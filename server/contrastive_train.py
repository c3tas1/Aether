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
# Import MiniBatchKMeans and PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA # <-- Added PCA
import argparse
import warnings
import time

# --- Environment Variable Setting (Optional - try if other things fail) ---
# Set BEFORE importing numpy/sklearn to potentially avoid threading issues
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# print("--- Set NUM_THREADS=1 ---") # Confirm if you uncomment these

warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# --- Configuration ---
parser = argparse.ArgumentParser(description='Unsupervised Image Classification using SimCLR, PCA, and MiniBatchKMeans')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for SimCLR training')
parser.add_argument('--epochs', type=int, default=50, help='Number of SimCLR training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for SimCLR')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for NT-Xent loss')
parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the projection head output')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
parser.add_argument('--image_size', type=int, default=224, help='Image size to resize to')
# --- PCA Argument ---
parser.add_argument('--pca_components', type=int, default=128, help='Number of dimensions to reduce to using PCA before clustering. Set to 0 to disable PCA.')
# --- KMeans Arguments ---
parser.add_argument('--kmeans_batch_size', type=int, default=2048, help='Mini-batch size for MiniBatchKMeans clustering')
parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters for K-Means (expected classes)')
parser.add_argument('--output_clusters_file', type=str, default='image_clusters.csv', help='CSV file to save image paths and cluster assignments')
parser.add_argument('--model_save_path', type=str, default=None, help='Optional path to save the trained SimCLR encoder model state dict')
parser.add_argument('--load_model_path', type=str, default=None, help='Optional path to load a pre-trained SimCLR model state dict (skips training)')

args = parser.parse_args()
resnet_output_dim = 2048 # Hardcoded for ResNet50

# (Keep Augmentations, Dataset Classes, SimCLRModel, NTXentLoss, train function the same as before)
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
    # (Same as before)
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
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)


class ImageListDataset(Dataset):
    # (Same as before)
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
            return None, img_path

def collate_fn_skip_corrupted(batch):
    # (Same as before)
    valid_batch = [(view, path) for view, path in batch if view is not None]
    if not valid_batch:
        return None, [path for _, path in batch]
    views = torch.utils.data.dataloader.default_collate([item[0] for item in valid_batch])
    paths = [item[1] for item in valid_batch]
    return views, paths

# --- Model (SimCLR Architecture) ---
class SimCLRModel(nn.Module):
    # (Same as before)
    def __init__(self, base_encoder_name='resnet50', projection_dim=128):
        super().__init__()
        if base_encoder_name == 'resnet50':
             base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
             base_output_dim = base_model.fc.in_features # 2048
        elif base_encoder_name == 'resnet18':
             base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
             base_output_dim = base_model.fc.in_features # 512
        else:
            raise ValueError(f"Unsupported base_encoder_name: {base_encoder_name}")
        self.encoder = base_model
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(base_output_dim, base_output_dim // 2),
            nn.ReLU(),
            nn.Linear(base_output_dim // 2, projection_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# --- NT-Xent Loss ---
class NTXentLoss(nn.Module):
    # (Same as before)
    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_mask(self, current_batch_size):
        mask = torch.ones((current_batch_size * 2, current_batch_size * 2), dtype=torch.bool).fill_diagonal_(0)
        for i in range(current_batch_size):
            mask[i, current_batch_size + i] = False
            mask[current_batch_size + i, i] = False
        return mask.to(self.device)

    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]
        if current_batch_size < 2:
             return torch.tensor(0.0, device=self.device, requires_grad=True)
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
        loss /= (2 * current_batch_size)
        return loss

# --- Training Function ---
def train(model, data_loader, optimizer, loss_fn, device, epochs):
    # (Same as before, including NaN check)
    model.train()
    total_loss = 0.0
    print("Starting SimCLR training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or len(batch_data) != 2: continue
            view1, view2 = batch_data
            view1, view2 = view1.to(device), view2.to(device)
            current_batch_size = view1.shape[0]
            if current_batch_size < 2: continue
            optimizer.zero_grad()
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = loss_fn(z1, z2)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        loader_len = len(data_loader)
        avg_epoch_loss = epoch_loss / loader_len if loader_len > 0 else 0.0
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss
        if args.model_save_path and (epoch + 1) % 10 == 0:
            save_path = f"{os.path.splitext(args.model_save_path)[0]}_epoch_{epoch+1}.pth"
            try:
                torch.save(model.encoder.state_dict(), save_path)
                print(f"Encoder model saved to {save_path}")
            except Exception as e:
                 print(f"Error saving checkpoint: {e}")

    print("Training finished.")
    return model

# --- Feature Extraction and Clustering (UPDATED) ---
def extract_features_and_cluster(model_encoder, root_dir, transform, device, num_clusters, mini_batch_size_kmeans, pca_components):
    print("\nExtracting features for clustering...")
    model_encoder.eval()
    model_encoder = model_encoder.to(device)

    dataset = ImageListDataset(root_dir, transform)
    eval_batch_size = args.batch_size * 2
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_corrupted)

    features_list = []
    image_paths = []

    with torch.no_grad():
        for batch_data, batch_paths in tqdm(data_loader, desc="Extracting Features"):
             if batch_data is None: continue
             batch_data = batch_data.to(device)
             batch_features = model_encoder(batch_data)
             features_list.append(batch_features.cpu().numpy())
             image_paths.extend(batch_paths)

    if not features_list:
        print("Error: No features were extracted.")
        return None, None

    # --- Concatenate Features ---
    print("Concatenating features...")
    try:
        features = np.concatenate(features_list, axis=0)
        # Ensure features are float32 (common source of unexpected memory usage if they become float64)
        if features.dtype != np.float32:
             print(f"Warning: Features array dtype is {features.dtype}. Converting to float32.")
             features = features.astype(np.float32)
    except MemoryError:
         print("ERROR: MemoryError during feature concatenation!")
         print(f"The full feature array ({len(image_paths)} x {resnet_output_dim}) likely does not fit in RAM.")
         print("Try reducing the dataset size or using IncrementalPCA (requires code modification).")
         return None, None
    except Exception as e:
         print(f"Error during feature concatenation: {e}")
         return None, None

    del features_list # Free memory

    print(f"Successfully extracted {features.shape[0]} feature vectors with dimension {features.shape[1]}.")
    print(f"Feature array size: {features.nbytes / (1024**2):.2f} MB") # Print actual size


    # --- Feature Validity Check ---
    print("Checking for non-finite values in features array...")
    if not np.all(np.isfinite(features)):
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        print(f"ERROR: Non-finite values detected! NaN count: {nan_count}, Inf count: {inf_count}")
        print("Cannot proceed with clustering. Check the model training or data.")
        # Optionally save the problematic features array for debugging
        # np.save("error_features.npy", features)
        return None, None
    else:
        print("Features array contains only finite values.")


    # --- Apply PCA (Optional) ---
    if pca_components > 0 and pca_components < features.shape[1]:
        print(f"\nApplying PCA to reduce dimensions from {features.shape[1]} to {pca_components}...")
        pca_start_time = time.time()
        try:
            # Using standard PCA - requires full features in memory.
            # If this step fails with MemoryError, IncrementalPCA would be needed.
            pca = PCA(n_components=pca_components, random_state=42)
            features_reduced = pca.fit_transform(features)
            pca_end_time = time.time()
            print(f"PCA finished in {pca_end_time - pca_start_time:.2f} seconds.")
            print(f"Reduced feature dimension: {features_reduced.shape[1]}.")
            print(f"Reduced feature array size: {features_reduced.nbytes / (1024**2):.2f} MB")
            # Replace original features with reduced ones
            features_to_cluster = features_reduced
            del features # Free memory of original full-dim features
        except MemoryError:
             print("ERROR: MemoryError during PCA fitting!")
             print("The dataset might be too large even for standard PCA.")
             print("Consider using IncrementalPCA (requires code modification) or reducing PCA components.")
             return None, None
        except Exception as e:
            print(f"Error during PCA: {e}")
            return None, None
    elif pca_components > 0:
         print(f"\nWarning: pca_components ({pca_components}) >= original dimensions ({features.shape[1]}). Skipping PCA.")
         features_to_cluster = features
    else:
        print("\nPCA disabled.")
        features_to_cluster = features


    # --- Use MiniBatchKMeans on potentially reduced features ---
    if features_to_cluster is None: # Check if PCA failed
         print("Error: Features to cluster are not available.")
         return None, None

    print(f"\nStarting MiniBatchKMeans clustering (k={num_clusters}, batch_size={mini_batch_size_kmeans})...")
    kmeans_start_time = time.time()
    mbkmeans = MiniBatchKMeans(n_clusters=num_clusters,
                               random_state=42,
                               batch_size=mini_batch_size_kmeans,
                               n_init=10,
                               max_iter=300,
                               reassignment_ratio=0.01,
                               verbose=0) # Set to 1 for more K-Means output

    try:
        # Fit MiniBatchKMeans
        mbkmeans.fit(features_to_cluster) # Fit on original or reduced features
        cluster_assignments = mbkmeans.labels_
        kmeans_end_time = time.time()
        print(f"Clustering finished in {kmeans_end_time - kmeans_start_time:.2f} seconds.")

        if len(image_paths) != len(cluster_assignments):
             print(f"CRITICAL WARNING: Mismatch between image paths ({len(image_paths)}) and assignments ({len(cluster_assignments)}). Results may be incorrect!")
        return image_paths, cluster_assignments

    except MemoryError:
         print("\nERROR: MemoryError during MiniBatchKMeans clustering!")
         print("Even with mini-batches, the process requires too much RAM.")
         print(f"Try reducing --kmeans_batch_size (currently {mini_batch_size_kmeans}) further.")
         print("If PCA was not used or components are high, try enabling/reducing --pca_components.")
         print("Ensure you have enough free system RAM and potentially add swap space.")
         return None, None
    except Exception as e:
        print(f"\nError during MiniBatchKMeans clustering: {e}")
        # Catching potential BLAS/LAPACK errors implicitly here too
        print("This might indicate issues with the data, memory limits, or underlying numerical libraries.")
        return None, None
    finally:
         # Explicitly delete large arrays no longer needed
         del features_to_cluster
         if 'features' in locals() and features_to_cluster is not features: # Delete original if PCA was used
              del features


# --- Main Execution ---
if __name__ == "__main__":
    start_total_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========================================")
    print(f"Starting Unsupervised Classification")
    print(f"Using device: {device}")
    print(f"Image Folder: {args.image_folder}")
    print(f"Parameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}, Temp={args.temperature}")
    print(f"PCA Components: {'Disabled' if args.pca_components <= 0 else args.pca_components}")
    print(f"Clustering: K={args.n_clusters}, KMeans Batch Size={args.kmeans_batch_size}")
    print(f"========================================")

    print("Initializing SimCLR model...")
    simclr_model = SimCLRModel(base_encoder_name='resnet50', projection_dim=args.embedding_dim)
    simclr_model = simclr_model.to(device)
    trained_encoder = None

    # --- Load or Train ---
    if args.load_model_path:
        # (Same model loading logic as before)
        if os.path.exists(args.load_model_path):
             print(f"Loading pre-trained encoder weights from: {args.load_model_path}")
             try:
                 simclr_model.encoder.load_state_dict(torch.load(args.load_model_path, map_location=device))
                 trained_encoder = simclr_model.encoder
                 print("Pre-trained encoder loaded successfully. Skipping training.")
             except Exception as e:
                 print(f"Error loading model weights: {e}. Training from scratch.")
        else:
            print(f"Warning: Load model path specified but not found: {args.load_model_path}. Training from scratch.")

    if trained_encoder is None:
        # (Same training preparation and execution as before)
        print("Preparing training dataset...")
        train_dataset = ContrastiveImageDataset(args.image_folder, contrastive_transform)
        effective_batch_size = min(args.batch_size, len(train_dataset))
        if effective_batch_size < 2: raise ValueError(f"Dataset size ({len(train_dataset)}) too small for effective batch size ({effective_batch_size}).")
        if effective_batch_size != args.batch_size: print(f"Warning: Reducing training batch size from {args.batch_size} to {effective_batch_size}.")
        use_drop_last = len(train_dataset) > effective_batch_size
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=use_drop_last)
        optimizer = optim.Adam(simclr_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
        nt_xent_loss = NTXentLoss(batch_size=effective_batch_size, temperature=args.temperature, device=device)
        simclr_model = train(simclr_model, train_loader, optimizer, nt_xent_loss, device, args.epochs)
        trained_encoder = simclr_model.encoder
        if args.model_save_path:
            try:
                save_dir = os.path.dirname(args.model_save_path); os.makedirs(save_dir, exist_ok=True) if save_dir else None
                torch.save(trained_encoder.state_dict(), args.model_save_path)
                print(f"Trained encoder model saved to: {args.model_save_path}")
            except Exception as e: print(f"Error saving model: {e}")
    else:
        print("Using pre-loaded encoder.")

    # --- Extract, (Reduce), Cluster ---
    if trained_encoder:
        # Pass the pca_components argument
        image_paths, cluster_ids = extract_features_and_cluster(
            trained_encoder,
            args.image_folder,
            eval_transform,
            device,
            num_clusters=args.n_clusters,
            mini_batch_size_kmeans=args.kmeans_batch_size,
            pca_components=args.pca_components # <-- Pass PCA components arg
        )

        # --- Save Results ---
        if image_paths is not None and cluster_ids is not None:
            # (Same saving logic as before)
            print(f"\nSaving cluster assignments to {args.output_clusters_file}...")
            try:
                output_dir = os.path.dirname(args.output_clusters_file); os.makedirs(output_dir, exist_ok=True) if output_dir else None
                with open(args.output_clusters_file, 'w') as f:
                    f.write("ImagePath,ClusterID\n")
                    for img_path, cluster_id in zip(image_paths, cluster_ids):
                        safe_img_path = str(img_path).replace(',', '_')
                        f.write(f"{safe_img_path},{cluster_id}\n")
                print(f"Saved {len(image_paths)} results.")
                print("\n--- Interpretation ---")
                print(f"Clustering complete. Results saved to '{args.output_clusters_file}'.")
                print(f"Images assigned to {args.n_clusters} clusters (IDs 0 to {args.n_clusters-1}).")
                print(">>> IMPORTANT: Manually inspect images from each cluster ID <<<")
                print(">>> to determine which cluster corresponds to 'object' and which to 'noise'. <<<")
            except Exception as e: print(f"\nError saving results to CSV: {e}")
        else:
            print("\nClustering step failed or produced no results. Output file not saved.")
    else:
        print("\nError: No trained or loaded encoder available.")

    end_total_time = time.time()
    print(f"\nTotal execution time: {(end_total_time - start_total_time)/60:.2f} minutes.")
    print("========================================")

    # --- Final Diagnostics ---
    import resource # Check resource usage (Linux/macOS only)
    if 'resource' in locals():
         mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
         print(f"Peak memory usage (self): {mem_usage / 1024:.2f} MB (Linux/macOS only)")
