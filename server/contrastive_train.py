import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='PIL') # Suppress Pillow warnings

# --- Configuration ---
# Use argparse for better script usability
parser = argparse.ArgumentParser(description='Unsupervised Image Classification using SimCLR')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for NT-Xent loss')
parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the final embedding space for loss calculation')
parser.add_argument('--output_dim', type=int, default=2048, help='Output dimension of the ResNet backbone') # ResNet50 default
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
parser.add_argument('--image_size', type=int, default=224, help='Image size to resize to')
parser.add_argument('--output_clusters_file', type=str, default='image_clusters.csv', help='CSV file to save image paths and cluster assignments')
args = parser.parse_args()

# --- Augmentations ---
# Crucial for contrastive learning
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=args.image_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # Optional: Gaussian Blur
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * args.image_size) // 2 * 2 + 1)], p=0.5), # kernel size must be odd
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
])

# Minimal transform for feature extraction after training
eval_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Dataset ---
class ContrastiveImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not self.image_files:
             raise ValueError(f"No compatible image files found in {root_dir}")
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # Apply the transform twice to get two augmented views
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            # Return dummy data or skip by returning None and handling in collate_fn (more complex)
            # For simplicity, let's retry with a random image
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)


class ImageListDataset(Dataset):
    """Dataset for feature extraction after training (returns image path too)"""
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not self.image_files:
             raise ValueError(f"No compatible image files found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            view = self.transform(image)
            return view, img_path # Return image path for later mapping
        except Exception as e:
            print(f"Warning: Skipping corrupted image during evaluation {img_path}: {e}")
            # Return None or dummy data if an error occurs
            return None, img_path # Caller needs to handle None


def collate_fn_skip_corrupted(batch):
    """Collate function that filters out None results"""
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None, None # Or raise error if batch becomes empty
    return torch.utils.data.dataloader.default_collate([item[0] for item in batch]), [item[1] for item in batch]


# --- Model (SimCLR Architecture) ---
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim, base_output_dim):
        super().__init__()
        self.encoder = base_encoder
        # Replace the classifier layer of the base encoder
        self.encoder.fc = nn.Identity() # Remove final classification layer

        # Add projection head
        self.projection_head = nn.Sequential(
            nn.Linear(base_output_dim, base_output_dim // 2), # Reduce dimension
            nn.ReLU(),
            nn.Linear(base_output_dim // 2, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z # Return both base features (h) and projected features (z)

# --- NT-Xent Loss ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # Mask to remove positive samples from similarity matrix diagonal
        self.mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(0)
        # Correct mask: only compare each image's augmentations with others, not within the same image pair initially
        for i in range(batch_size):
            self.mask[i, batch_size + i] = False
            self.mask[batch_size + i, i] = False
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        z_i: representations of view 1 (batch_size x feature_dim)
        z_j: representations of view 2 (batch_size x feature_dim)
        """
        batch_size = z_i.shape[0] # Get actual batch size (might be smaller at the end)
        if batch_size != self.batch_size: # Adjust mask if last batch is smaller
            mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(0).to(self.device)
            for i in range(batch_size):
                mask[i, batch_size + i] = False
                mask[batch_size + i, i] = False
        else:
            mask = self.mask.to(self.device)


        z = torch.cat((z_i, z_j), dim=0) # (2*batch_size) x feature_dim
        z = F.normalize(z, p=2, dim=1) # L2 Normalize the representations

        # Calculate cosine similarity
        # sim[i, j] = similarity between z[i] and z[j]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # (2*batch_size) x (2*batch_size)

        # --- Create labels and masks for CrossEntropy ---
        # Positive pairs: (z_i[k], z_j[k]) for k=0..batch_size-1
        # Indices of positive pairs in the concatenated tensor 'z'
        pos_idx_1 = torch.arange(batch_size).to(self.device)
        pos_idx_2 = torch.arange(batch_size, 2 * batch_size).to(self.device)

        # Similarities corresponding to positive pairs
        sim_i_j = torch.diag(sim, diagonal=batch_size) # sim(z_i[k], z_j[k])
        sim_j_i = torch.diag(sim, diagonal=-batch_size) # sim(z_j[k], z_i[k])

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size, 1) # (2*batch_size) x 1

        # Select negative samples using the mask
        negative_samples = sim[mask].reshape(2 * batch_size, -1) # (2*batch_size) x (2*batch_size - 2)

        # Logits for CrossEntropy: [positive_sample, negative_samples_...]
        logits = torch.cat((positive_samples, negative_samples), dim=1) # (2*batch_size) x (2*batch_size - 1)

        # Labels: the positive sample is always at index 0
        labels = torch.zeros(2 * batch_size).to(self.device).long()

        # Calculate loss
        loss = self.criterion(logits, labels)
        loss /= (2 * batch_size) # Normalize loss by batch size
        return loss

# --- Training Function ---
def train(model, data_loader, optimizer, loss_fn, device, epochs):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (view1, view2) in enumerate(progress_bar):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # Get representations
            _, z1 = model(view1) # Use projected features for loss
            _, z2 = model(view2)

            # Handle potential smaller last batch in loss function
            current_batch_size = z1.shape[0]
            if current_batch_size < 2 : continue # Skip if batch too small for loss calc

            loss = loss_fn(z1, z2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss

        # Optional: Save checkpoints
        # torch.save(model.state_dict(), f"simclr_model_epoch_{epoch+1}.pth")

    print("Training finished.")
    return model # Return trained model

# --- Feature Extraction and Clustering ---
def extract_features_and_cluster(model, root_dir, transform, device, num_clusters=2):
    print("\nExtracting features...")
    model.eval() # Set model to evaluation mode
    # We only need the encoder part now
    encoder = model.encoder
    encoder = encoder.to(device)

    dataset = ImageListDataset(root_dir, transform)
    # Use collate_fn to handle potential image loading errors during eval
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_corrupted)

    features = []
    image_paths = []

    with torch.no_grad():
        for batch_data, batch_paths in tqdm(data_loader, desc="Extracting Features"):
             if batch_data is None: continue # Skip if batch was empty after filtering Nones
             batch_data = batch_data.to(device)
             batch_features = encoder(batch_data) # Get features from the encoder only
             features.append(batch_features.cpu().numpy())
             image_paths.extend(batch_paths) # Keep track of original image paths

    if not features:
        print("Error: No features were extracted. Check image loading and dataset.")
        return None, None

    features = np.concatenate(features, axis=0)
    print(f"Extracted {features.shape[0]} feature vectors with dimension {features.shape[1]}.")
    print("Starting K-Means clustering...")

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # n_init to improve stability
    cluster_assignments = kmeans.fit_predict(features)

    print("Clustering finished.")
    return image_paths, cluster_assignments

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Prepare Data ---
    train_dataset = ContrastiveImageDataset(args.image_folder, contrastive_transform)
    # Adjust batch size if dataset is smaller
    effective_batch_size = min(args.batch_size, len(train_dataset))
    if effective_batch_size < 2:
         raise ValueError(f"Dataset size ({len(train_dataset)}) is too small for the batch size ({args.batch_size}) required for contrastive loss.")

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True) # drop_last=True ensures consistent batch size for loss

    # --- Prepare Model ---
    # Load a pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)
    # Get the output dimension of the layer before the final FC layer
    resnet_output_dim = resnet.fc.in_features # Usually 2048 for ResNet50

    simclr_model = SimCLRModel(base_encoder=resnet, projection_dim=args.embedding_dim, base_output_dim=resnet_output_dim)
    simclr_model = simclr_model.to(device)

    # --- Prepare Optimizer and Loss ---
    optimizer = optim.Adam(simclr_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    # Note: SimCLR often uses LARS optimizer, but Adam is generally fine
    nt_xent_loss = NTXentLoss(batch_size=effective_batch_size, temperature=args.temperature, device=device)

    # --- Train Model ---
    print("Starting training...")
    trained_model = train(simclr_model, train_loader, optimizer, nt_xent_loss, device, args.epochs)

    # --- Extract Features and Cluster ---
    image_paths, cluster_ids = extract_features_and_cluster(trained_model, args.image_folder, eval_transform, device, num_clusters=2)

    # --- Save Results ---
    if image_paths is not None and cluster_ids is not None:
        print(f"\nSaving cluster assignments to {args.output_clusters_file}...")
        with open(args.output_clusters_file, 'w') as f:
            f.write("ImagePath,ClusterID\n")
            for img_path, cluster_id in zip(image_paths, cluster_ids):
                f.write(f"{img_path},{cluster_id}\n")
        print("Done.")

        # --- Interpretation ---
        print("\n--- Interpretation ---")
        print(f"Clustering complete. Results saved to '{args.output_clusters_file}'.")
        print("The images have been assigned to two clusters (0 and 1).")
        print("You should now MANUALLY INSPECT images from each cluster ID to determine which cluster corresponds to 'object' and which corresponds to 'noise'.")
        print("For example, look at a few images listed with ClusterID=0 and a few with ClusterID=1 in the CSV file.")
