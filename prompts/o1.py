import os
import re
import random
import argparse
from collections import defaultdict
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.models as models  # For CNN-based example
# from torchvision.models import vit_b_16  # Example for Vision Transformer

# -----------------------
# 1) Argument Parser
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the root directory containing class subfolders.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="Local rank for distributed training.")
    return parser.parse_args()


# -----------------------
# 2) Custom Dataset
# -----------------------
class ImageDataset(Dataset):
    """
    A simple dataset that expects a list of (image_path, label) tuples.
    Applies transforms and returns (image_tensor, label).
    """

    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, class_index)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert("RGB")


# -----------------------
# 3) Helper Functions
# -----------------------
def find_image_id_and_patch(filename):
    """
    Given a filename of the format:
        ClassName_{ImageID}_Patch_{PatchNumber}.jpg
    Return (ClassName, ImageID, PatchNumber).
    
    For example:
        "Class_0_Class0Image1_Patch_2.jpg" -> ("Class_0", "Class0Image1", 2)
    We'll rely on capturing logic with a regex or manual split.
    """
    # Example pattern: Class_0_Class0Image1_Patch_2.jpg
    # Let's do a naive split approach:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_Patch_")
    patch_num = parts[-1]
    prefix = parts[0]
    
    # prefix might be something like: Class_0_Class0Image1
    # We can find the first underscore to get the class name:
    # but be careful with classes that might contain underscores themselves.
    # If classes are truly "Class_0", we can do:
    #   class_name, image_id = prefix.split("_", 1)
    # But let's assume class_name is the first underscore portion:
    # It's safer to pass in the class name from the folder structure, though.
    # For demonstration, let's do:
    match = re.match(r"(.*?)_(.*)", prefix)
    if match:
        class_name = match.group(1)  # "Class_0"
        image_id = match.group(2)   # "Class0Image1"
    else:
        # fallback, if pattern doesn't match
        class_name = "Unknown"
        image_id = prefix
    
    return class_name, image_id, int(patch_num)


def gather_dataset_info(data_dir):
    """
    Parse each class subfolder, gather all .jpg files, 
    and group them by (class_name, image_id).
    Returns:
        data_dict = {
           class_name: {
               image_id1: [file1, file2, ...],
               image_id2: [...],
               ...
           },
           ...
        }
    Also returns the total count per class.
    """
    data_dict = defaultdict(lambda: defaultdict(list))
    class_counts = defaultdict(int)

    # List all class subfolders in data_dir
    class_folders = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]

    for cls in class_folders:
        class_folder = os.path.join(data_dir, cls)
        # gather all jpg files
        image_files = glob(os.path.join(class_folder, "*.jpg"))
        for img_file in image_files:
            # parse out image_id
            # Optionally validate that `cls` matches the class_name in the file
            # But typically just use the subfolder name as ground truth
            _, image_id, _ = find_image_id_and_patch(img_file)
            data_dict[cls][image_id].append(img_file)
        # After collecting, count total
        count = sum(len(img_list) for img_list in data_dict[cls].values())
        class_counts[cls] = count

    return data_dict, class_counts


def split_train_val(data_dict, class_counts, min_count=5000, max_count=10000, train_ratio=0.8):
    """
    1) Omit classes with < min_count images.
    2) For classes with > max_count images, limit to max_count.
    3) Split each class by image_id in an 80:20 ratio (train:val).
       ensuring entire sets of patches from one image_id remain in the same split.
    Returns:
        train_list = [(filepath, class_index), ...]
        val_list = [(filepath, class_index), ...]
        valid_classes = sorted list of classes that we actually include
        class_to_weight = dictionary with final weight to apply in loss
    """
    train_list = []
    val_list = []
    valid_classes = []
    class_to_idx = {}
    class_to_weight = {}

    # We only keep classes that pass the threshold
    # We'll create a sorted list to ensure consistent indexing
    # and map classes -> indices
    included_classes = []
    for cls, count in class_counts.items():
        if count < min_count:
            continue
        included_classes.append(cls)
    included_classes.sort()

    for i, cls in enumerate(included_classes):
        class_to_idx[cls] = i

    # Compute weights for the classes that pass the threshold
    # If 5k <= count <= 10k, apply some weight > 1
    # If count > 10k, treat as weight=1 but limit images to 10k
    # (You can adjust weighting logic as desired.)
    for cls in included_classes:
        count = class_counts[cls]
        if count <= max_count:
            # 5k .. 10k
            class_to_weight[cls] = 2.0  # example weighting
        else:
            # > 10k
            class_to_weight[cls] = 1.0

    # Build train/val splits
    for cls in included_classes:
        cls_index = class_to_idx[cls]
        # List all image_ids for this class
        image_id_list = list(data_dict[cls].keys())
        # If class has more than max_count images, we limit to max_count
        # We must do this at the "image" group level, so let's randomize the image_id_list
        random.shuffle(image_id_list)
        
        # Flatten the patch list to see total
        total_patches = sum(len(data_dict[cls][img_id]) for img_id in image_id_list)
        # We'll pick images (and their patches) until we reach max_count (if needed).
        
        # We need to accumulate patches but keep images intact.
        selected_image_ids = []
        running_count = 0
        for img_id in image_id_list:
            patch_count = len(data_dict[cls][img_id])
            if running_count + patch_count <= max_count:
                selected_image_ids.append(img_id)
                running_count += patch_count
            else:
                # If adding this image_id's patches would exceed max_count,
                # we can either skip it or partially include it. 
                # But we must keep patches for an image_id together,
                # so let's skip if we exceed the limit.
                break

        # Now we have a list of selected_image_ids (with total <= max_count).
        # Perform 80:20 split by number of image_ids
        # so that all patches of a given image_id are in the same set.
        split_index = int(len(selected_image_ids) * train_ratio)
        train_img_ids = selected_image_ids[:split_index]
        val_img_ids = selected_image_ids[split_index:]

        # Build final train/val lists
        for tid in train_img_ids:
            for f in data_dict[cls][tid]:
                train_list.append((f, cls_index))
        for vid in val_img_ids:
            for f in data_dict[cls][vid]:
                val_list.append((f, cls_index))

        valid_classes.append(cls)

    return train_list, val_list, valid_classes, class_to_idx, class_to_weight


# -----------------------
# 4) Main training logic
# -----------------------
def main_worker(local_rank, world_size, args):
    # -----------------------
    # 4.1) Initialize Process Group
    # -----------------------
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # -----------------------
    # 4.2) Prepare Dataset
    # -----------------------
    data_dict, class_counts = gather_dataset_info(args.data_dir)

    train_list, val_list, valid_classes, class_to_idx, class_to_weight = split_train_val(
        data_dict, class_counts, min_count=5000, max_count=10000, train_ratio=0.8
    )

    num_classes = len(valid_classes)

    # Create transform (example)
    # You can add stronger augmentations for real tasks
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # Create Dataset objects
    train_dataset = ImageDataset(train_list, transform=train_transform)
    val_dataset = ImageDataset(val_list, transform=val_transform)

    # -----------------------
    # 4.3) Create Distributed Samplers and Loaders
    # -----------------------
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=8,         # Adjust based on your system
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True
    )

    # -----------------------
    # 4.4) Build Model
    # -----------------------
    # Example: Use a ResNet or a Vision Transformer
    # model = vit_b_16(pretrained=True)  # For Vision Transformer from torchvision >= 0.13
    # model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # -----------------------
    # 4.5) Create Optimizer, Criterion
    # -----------------------
    # Build class weights tensor for CrossEntropyLoss
    # According to class_to_weight
    weights = []
    # valid_classes is sorted, so class_to_idx[cls] is consistent
    for cls in valid_classes:
        weights.append(class_to_weight[cls])
    weight_tensor = torch.FloatTensor(weights).cuda(local_rank)

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=0.1  # Example label smoothing for noisy labels
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------
    # 4.6) Training Loop
    # -----------------------
    for epoch in range(args.epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and local_rank == 0:
                print(f"[Epoch {epoch} Iter {i}] loss: {loss.item():.4f}")

        # Average training loss across all steps
        train_loss = running_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda(local_rank, non_blocking=True)
                labels = labels.cuda(local_rank, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total

        # Print metrics (only from rank 0 to avoid duplicates)
        if local_rank == 0:
            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Acc: {accuracy:.2f}%")

    # Cleanup
    dist.destroy_process_group()


def main():
    args = parse_args()
    
    # We assume you launch with something like:
    #   torchrun --nproc_per_node=8 train.py --data_dir=... 
    # So world_size is the number of GPUs available across nodes
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    local_rank = args.local_rank

    # For PyTorch Distributed, local_rank is automatically set if using 
    # torchrun or torch.distributed.launch. We'll pass it into main_worker.
    mp.set_start_method("spawn", force=True)

    main_worker(local_rank, world_size, args)


if __name__ == "__main__":
    main()

