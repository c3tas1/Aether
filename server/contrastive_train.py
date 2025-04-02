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
# Import MiniBatchKMeans and remove KMeans
from sklearn.cluster import MiniBatchKMeans
import argparse
import warnings
import time # To potentially add timing info

warnings.filterwarnings("ignore", category=UserWarning, module='PIL') # Suppress Pillow warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn') # Suppress future warnings from sklearn if any

# --- Configuration ---
parser = argparse.ArgumentParser(description='Unsupervised Image Classification using SimCLR and MiniBatchKMeans')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for SimCLR training')
parser.add_argument('--epochs', type=int, default=50, help='Number of SimCLR training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for SimCLR')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for NT-Xent loss')
parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the projection head output')
# ResNet50 output is 2048, let's keep this fixed for clarity unless backbone changes
# parser.add_argument('--output_dim', type=int, default=2048, help='Output dimension of the ResNet backbone')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
parser.add_argument('--image_size', type=int, default=224, help='Image size to resize to')
parser.add_argument('--kmeans_batch_size', type=int, default=2048, help='Mini-batch size for MiniBatchKMeans clustering')
parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters for K-Means (expected classes)')
parser.add_argument('--output_clusters_file', type=str, default='image_clusters.csv', help='CSV file to save image paths and cluster assignments')
parser.add_argument('--model_save_path', type=str, default=None, help='Optional path to save the trained SimCLR encoder model state dict')
parser.add_argument('--load_model_path', type=str, default=None, help='Optional path to load a pre-trained SimCLR model state dict (skips training)')

args = parser.parse_args()
resnet_output_dim = 2048 # Hardcoded for ResNet50

# --- Augmentations ---
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=args.image_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=max(1, int(0.1 * args.image_size) // 2 * 2 + 1))], p=0.5), # kernel size must be odd and >=1
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        self.image_files = []
        print(f"Searching for images in: {root_dir}")
        for f in os.listdir(root_dir):
            f_path = os.path.join(root_dir, f)
            if os.path.isfile(f_path) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                 self.image_files.append(f_path)

        if not self.image_files:
             raise ValueError(f"No compatible image files found in {root_dir}")
        print(f"Found {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        except Exception as e:
            print(f"\nWarning: Error loading image {img_path}, returning dummy data or retrying: {e}")
            # Simple retry mechanism: try loading the next image index
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)


class ImageListDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        if not self.image_files:
             raise ValueError(f"No compatible image files found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            view = self.transform(image)
            return view, img_path
        except Exception as e:
            print(f"\nWarning: Skipping corrupted image during evaluation {img_path}: {e}")
            return None, img_path # Return None for image data

def collate_fn_skip_corrupted(batch):
    """Collate function that filters out None image data but keeps paths."""
    valid_batch = [(view, path) for view, path in batch if view is not None]
    if not valid_batch:
        # Return paths even if all images in batch failed
        return None, [path for _, path in batch]

    # Use default collate for valid views
    views = torch.utils.data.dataloader.default_collate([item[0] for item in valid_batch])
    paths = [item[1] for item in valid_batch] # Paths corresponding to valid views
    all_paths_in_batch = [item[1] for item in batch] # All paths originally in batch

    # Need a way to map results back later, maybe return all paths and indices of valid ones?
    # Simpler: just return valid views and their corresponding paths for now.
    # Clustering will only happen on successfully extracted features.
    return views, paths


# --- Model (SimCLR Architecture) ---
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder_name='resnet50', projection_dim=128):
        super().__init__()
        if base_encoder_name == 'resnet50':
             base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
             base_output_dim = base_model.fc.in_features # 2048
        elif base_encoder_name == 'resnet18':
             base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
             base_output_dim = base_model.fc.in_features # 512
        # Add other backbones if needed
        else:
            raise ValueError(f"Unsupported base_encoder_name: {base_encoder_name}")

        self.encoder = base_model
        self.encoder.fc = nn.Identity() # Remove final classification layer

        self.projection_head = nn.Sequential(
            nn.Linear(base_output_dim, base_output_dim // 2),
            nn.ReLU(),
            nn.Linear(base_output_dim // 2, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z # h: encoder output (for clustering), z: projection output (for loss)

# --- NT-Xent Loss ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # Use sum reduction

    def _get_mask(self, current_batch_size):
        """Generate mask dynamically based on current batch size."""
        mask = torch.ones((current_batch_size * 2, current_batch_size * 2), dtype=torch.bool).fill_diagonal_(0)
        for i in range(current_batch_size):
            mask[i, current_batch_size + i] = False
            mask[current_batch_size + i, i] = False
        return mask.to(self.device)

    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]
        if current_batch_size < 2: # Need at least 2 samples for pairs
             print(f"Warning: Skipping loss calculation for batch size {current_batch_size}")
             return torch.tensor(0.0, device=self.device, requires_grad=True) # Return zero loss

        mask = self._get_mask(current_batch_size)

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, p=2, dim=1)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, diagonal=current_batch_size)
        sim_j_i = torch.diag(sim, diagonal=-current_batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * current_batch_size, 1)
        negative_samples = sim[mask].reshape(2 * current_batch_size, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(2 * current_batch_size).to(self.device).long()

        loss = self.criterion(logits, labels)
        # Normalize by the number of samples (2 * batch_size) as per SimCLR paper appendix B
        loss /= (2 * current_batch_size)
        return loss

# --- Training Function ---
def train(model, data_loader, optimizer, loss_fn, device, epochs):
    model.train()
    total_loss = 0.0
    print("Starting SimCLR training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx, batch_data in enumerate(progress_bar):
            # Check if batch data is valid (can be None if all images failed in collate_fn)
            if batch_data is None or len(batch_data) != 2:
                 print(f"Warning: Skipping empty or invalid batch {batch_idx}")
                 continue
            view1, view2 = batch_data
            # Ensure tensors are on the correct device
            view1, view2 = view1.to(device), view2.to(device)

            # Check batch size again after moving to device, just in case
            current_batch_size = view1.shape[0]
            if current_batch_size < 2:
                # print(f"Warning: Skipping batch {batch_idx} with size {current_batch_size} < 2")
                continue

            optimizer.zero_grad()

            _, z1 = model(view1) # Projected features for loss
            _, z2 = model(view2)

            loss = loss_fn(z1, z2)

            # Check if loss is NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                # Optionally add more debugging here (e.g., print norms of z1, z2)
                continue # Skip optimizer step if loss is invalid


            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Calculate average loss avoiding division by zero if loader is empty
        loader_len = len(data_loader)
        avg_epoch_loss = epoch_loss / loader_len if loader_len > 0 else 0.0
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss

        # Optional: Save checkpoints
        if args.model_save_path and (epoch + 1) % 10 == 0: # Save every 10 epochs
            save_path = f"{os.path.splitext(args.model_save_path)[0]}_epoch_{epoch+1}.pth"
            # Save only the encoder state_dict for clustering later
            torch.save(model.encoder.state_dict(), save_path)
            print(f"Encoder model saved to {save_path}")

    print("Training finished.")
    return model

# --- Feature Extraction and Clustering ---
def extract_features_and_cluster(model_encoder, root_dir, transform, device, num_clusters, mini_batch_size_kmeans):
    print("\nExtracting features for clustering...")
    model_encoder.eval() # Ensure encoder is in eval mode
    model_encoder = model_encoder.to(device) # Move encoder to device

    dataset = ImageListDataset(root_dir, transform)
    # Use a reasonable batch size for feature extraction; can be larger than training batch
    eval_batch_size = args.batch_size * 2
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_corrupted)

    features_list = []
    image_paths = [] # Store paths corresponding to successfully extracted features

    with torch.no_grad():
        for batch_data, batch_paths in tqdm(data_loader, desc="Extracting Features"):
             # Skip if the batch processing failed in collate_fn
             if batch_data is None:
                 print(f"Warning: Skipping a batch during feature extraction due to loading errors.")
                 continue
             batch_data = batch_data.to(device)
             batch_features = model_encoder(batch_data) # Get features from the encoder only
             features_list.append(batch_features.cpu().numpy())
             image_paths.extend(batch_paths) # Add paths for features successfully extracted

    if not features_list:
        print("Error: No features were extracted. Cannot perform clustering. Check image loading.")
        return None, None

    features = np.concatenate(features_list, axis=0)
    del features_list # Free memory
    print(f"Successfully extracted {features.shape[0]} feature vectors with dimension {features.shape[1]}.")

    # --- Use MiniBatchKMeans ---
    print(f"Starting MiniBatchKMeans clustering (k={num_clusters}, batch_size={mini_batch_size_kmeans})...")
    start_time = time.time()
    # Use MiniBatchKMeans for memory efficiency
    mbkmeans = MiniBatchKMeans(n_clusters=num_clusters,
                               random_state=42,
                               batch_size=mini_batch_size_kmeans,
                               n_init=10, # Run 10 initializations, common default. Reduce if memory/time constrained.
                               max_iter=300, # Maximum iterations per initialization
                               reassignment_ratio=0.01, # Helps convergence for large datasets
                               verbose=0) # Set to 1 or higher for more output

    # Fit MiniBatchKMeans
    try:
        mbkmeans.fit(features)
        cluster_assignments = mbkmeans.labels_
        end_time = time.time()
        print(f"Clustering finished in {end_time - start_time:.2f} seconds.")
        # Ensure we have assignments for all images we got features for
        if len(image_paths) != len(cluster_assignments):
             print(f"Warning: Mismatch between number of image paths ({len(image_paths)}) and cluster assignments ({len(cluster_assignments)}). This should not happen.")
             # Attempt to proceed, but results might be misaligned
        return image_paths, cluster_assignments

    except Exception as e:
        print(f"\nError during MiniBatchKMeans clustering: {e}")
        print("This might be due to memory limits even with mini-batches, or issues with the feature data.")
        print(f"Try reducing --kmeans_batch_size (currently {mini_batch_size_kmeans}) or consider applying PCA first if dimensions are very high.")
        return None, None


# --- Main Execution ---
if __name__ == "__main__":
    start_total_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========================================")
    print(f"Starting Unsupervised Classification")
    print(f"Using device: {device}")
    print(f"Image Folder: {args.image_folder}")
    print(f"Parameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}, Temp={args.temperature}")
    print(f"Clustering: K={args.n_clusters}, KMeans Batch Size={args.kmeans_batch_size}")
    print(f"========================================")


    # --- Prepare Model ---
    print("Initializing SimCLR model...")
    # Using resnet50 as default, could make this an argument too
    simclr_model = SimCLRModel(base_encoder_name='resnet50', projection_dim=args.embedding_dim)
    simclr_model = simclr_model.to(device)

    trained_encoder = None

    if args.load_model_path:
        if os.path.exists(args.load_model_path):
             print(f"Loading pre-trained encoder weights from: {args.load_model_path}")
             try:
                 # Load state dict for the encoder part only
                 simclr_model.encoder.load_state_dict(torch.load(args.load_model_path, map_location=device))
                 trained_encoder = simclr_model.encoder # Use the loaded encoder
                 print("Pre-trained encoder loaded successfully. Skipping training.")
             except Exception as e:
                 print(f"Error loading model weights: {e}. Proceeding to train from scratch.")
                 trained_encoder = None # Ensure training happens if loading fails
        else:
            print(f"Warning: Load model path specified but not found: {args.load_model_path}. Training from scratch.")

    # --- Train Model (if not loaded) ---
    if trained_encoder is None:
        # --- Prepare Data ---
        print("Preparing training dataset...")
        train_dataset = ContrastiveImageDataset(args.image_folder, contrastive_transform)
        effective_batch_size = min(args.batch_size, len(train_dataset))
        if effective_batch_size < 2:
             raise ValueError(f"Dataset size ({len(train_dataset)}) is too small for the effective batch size ({effective_batch_size}) required for contrastive loss.")
        if effective_batch_size != args.batch_size:
             print(f"Warning: Reducing training batch size from {args.batch_size} to {effective_batch_size} due to small dataset size.")

        # Use drop_last=True for stable batch size in loss, unless dataset is very small
        use_drop_last = len(train_dataset) > effective_batch_size
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=use_drop_last)

        # --- Prepare Optimizer and Loss ---
        optimizer = optim.Adam(simclr_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
        nt_xent_loss = NTXentLoss(batch_size=effective_batch_size, temperature=args.temperature, device=device)

        # --- Train ---
        simclr_model = train(simclr_model, train_loader, optimizer, nt_xent_loss, device, args.epochs)
        trained_encoder = simclr_model.encoder # Get the encoder part after training

        # --- Save Trained Model (Optional) ---
        if args.model_save_path:
            try:
                # Ensure directory exists if path includes directories
                save_dir = os.path.dirname(args.model_save_path)
                if save_dir: # If path is not just a filename
                    os.makedirs(save_dir, exist_ok=True)
                # Save only the encoder's state dictionary
                torch.save(trained_encoder.state_dict(), args.model_save_path)
                print(f"Trained encoder model state dictionary saved to: {args.model_save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
    else:
        print("Using pre-loaded encoder.")


    # --- Extract Features and Cluster ---
    if trained_encoder: # Proceed only if we have an encoder (either trained or loaded)
        image_paths, cluster_ids = extract_features_and_cluster(
            trained_encoder,
            args.image_folder,
            eval_transform,
            device,
            num_clusters=args.n_clusters,
            mini_batch_size_kmeans=args.kmeans_batch_size
        )

        # --- Save Results ---
        if image_paths is not None and cluster_ids is not None:
            print(f"\nSaving cluster assignments to {args.output_clusters_file}...")
            try:
                output_dir = os.path.dirname(args.output_clusters_file)
                if output_dir: os.makedirs(output_dir, exist_ok=True)

                with open(args.output_clusters_file, 'w') as f:
                    f.write("ImagePath,ClusterID\n")
                    for img_path, cluster_id in zip(image_paths, cluster_ids):
                        # Basic sanitation: replace potential commas in path
                        safe_img_path = str(img_path).replace(',', '_')
                        f.write(f"{safe_img_path},{cluster_id}\n")
                print(f"Saved {len(image_paths)} results.")

                 # --- Interpretation Guidance ---
                print("\n--- Interpretation ---")
                print(f"Clustering complete. Results saved to '{args.output_clusters_file}'.")
                print(f"Images assigned to {args.n_clusters} clusters (IDs 0 to {args.n_clusters-1}).")
                print(">>> IMPORTANT: Manually inspect images from each cluster ID in the CSV file <<<")
                print(">>> to determine which cluster corresponds to 'object' and which to 'noise'. <<<")

            except Exception as e:
                print(f"\nError saving results to CSV: {e}")
        else:
            print("\nClustering step failed or produced no results. Output file not saved.")
    else:
        print("\nError: No trained or loaded encoder available. Cannot proceed with feature extraction and clustering.")

    end_total_time = time.time()
    print(f"\nTotal execution time: {(end_total_time - start_total_time)/60:.2f} minutes.")
    print("========================================")
