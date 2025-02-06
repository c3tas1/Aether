import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# Configuration
DATASET_ROOT = '/path/to/dataset'
OUTPUT_DIR = '/path/to/output'
BATCH_SIZE = 512
NUM_EPOCHS = 100
LR = 0.001
NUM_WORKERS = 16
IMG_SIZE = 224

# Distributed training setup
dist.init_process_group('nccl')
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
device = torch.device('cuda')

def generate_splits():
    class_data = defaultdict(lambda: defaultdict(list))
    valid_classes = []
    
    for class_name in os.listdir(DATASET_ROOT):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        image_groups = defaultdict(list)
        for fn in os.listdir(class_dir):
            if fn.endswith('.jpg'):
                base = fn.rsplit('_patch_', 1)[0]
                image_groups[base].append(os.path.join(class_dir, fn))
        
        total_patches = sum(len(v) for v in image_groups.values())
        if total_patches < 5000:
            continue
        
        valid_classes.append(class_name)
        all_images = list(image_groups.items())
        
        if total_patches > 10000:
            random.shuffle(all_images)
            selected, count = [], 0
            for img_group in all_images:
                if count + len(img_group[1]) > 10000:
                    continue
                selected.append(img_group)
                count += len(img_group[1])
                if count >= 10000:
                    break
            all_images = selected
        
        random.shuffle(all_images)
        split_idx = int(0.8 * len(all_images))
        class_data[class_name]['train'] = [p for img in all_images[:split_idx] for p in img[1]]
        class_data[class_name]['val'] = [p for img in all_images[split_idx:] for p in img[1]]
    
    return class_data, valid_classes

class MultiLabelDataset(Dataset):
    def __init__(self, data, class_map, transform=None):
        self.samples = []
        self.class_map = class_map
        self.transform = transform
        
        for class_name in data:
            class_idx = self.class_map[class_name]
            self.samples.extend([(p, class_idx) for p in data[class_name]])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Generate dataset splits
class_data, valid_classes = generate_splits()
class_map = {cn: i for i, cn in enumerate(valid_classes)}

# Calculate class weights
class_counts = {class_map[cn]: len(class_data[cn]['train']) for cn in valid_classes}
weights = []
for cn in valid_classes:
    count = class_counts[class_map[cn]]
    if 5000 <= count <= 10000:
        weights.append(1 / (count ** 0.5))
    else:
        weights.append(1.0)
        
weights = torch.tensor(weights, device=device)
weights = weights / weights.sum() * len(weights)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = MultiLabelDataset(
    {cn: class_data[cn]['train'] for cn in valid_classes},
    class_map,
    train_transform
)

val_dataset = MultiLabelDataset(
    {cn: class_data[cn]['val'] for cn in valid_classes},
    class_map,
    val_transform
)

# Distributed samplers
train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

# Create loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

# Model (Vision Transformer)
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, len(valid_classes))
model = DDP(model.to(device), device_ids=[device])

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scaler = GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_sampler.set_epoch(epoch)
    
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    scheduler.step()
    
    # Validation
    if dist.get_rank() == 0 and epoch % 5 == 0:
        model.eval()
        total_loss, total_correct = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / len(val_dataset)
        print(f'Epoch {epoch+1}: Val Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}')

dist.destroy_process_group()

