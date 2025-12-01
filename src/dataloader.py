import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CausalTimeSeriesDataset(Dataset):
    """
    Professional time-series Dataset supporting sliding window slicing with low memory usage.
    """
    def __init__(self, data, window_size, stride=1, mode='train', split_ratio=0.8):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        
        # 1. Data splitting (train/validation split along time axis)
        split_point = int(len(data) * split_ratio)
        if mode == 'train':
            self.data = data[:split_point]
        elif mode == 'val':
            self.data = data[split_point:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Calculate total number of samples
        if len(self.data) < window_size:
            self.n_samples = 0
        else:
            self.n_samples = (len(self.data) - window_size) // stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end]
        # (T, N) -> (N, T)
        sample_tensor = torch.from_numpy(sample).float().t() 
        return sample_tensor

def load_from_disk(base_path, dataset_name, replica_id):
    """
    Load files from disk (Numpy format).
    """
    data_dir = os.path.join(base_path, dataset_name)
    data_path = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_path = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_path = os.path.join(data_dir, f'coords_{replica_id}.npy')

    # 1. Data is required
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data not found: {data_path}")
    data_np = np.load(data_path) # Shape: (T, N)
    N = data_np.shape[1]

    # 2. GT is optional
    if os.path.exists(gt_path):
        gt_np = np.load(gt_path) # Shape: (N, N)
    else:
        print(f"⚠️ Warning: Ground Truth not found at {gt_path}. Metrics will be skipped.")
        gt_np = None

    # 3. Coords are also optional
    if os.path.exists(coords_path):
        coords_np = np.load(coords_path) # Shape: (N, 2)
    else:
        print(f"⚠️ Warning: Coords not found at {coords_path}. Using random coordinates.")
        np.random.seed(42)
        coords_np = np.random.rand(N, 2)
    
    return data_np, gt_np, coords_np

def get_data_context(args):
    """
    Factory function: returns Train/Val Loaders and Meta
    """
    base_path = getattr(args, 'data_path', 'data/synthetic')
    dataset_name = getattr(args, 'dataset', 'lorenz96')
    replica_id = getattr(args, 'replica_id', 0)
    
    window_size = getattr(args, 'window_size', 100) 
    stride = getattr(args, 'stride', 10)
    batch_size = getattr(args, 'batch_size', 32)
    norm_coords = getattr(args, 'norm_coords', False)

    print(f"📂 Loading {dataset_name} (Replica {replica_id})...")
    
    data_np, gt_np, coords_np = load_from_disk(base_path, dataset_name, replica_id)

    # 1. Time-series data standardization (Z-Score) - always performed
    mean = data_np.mean(axis=0)
    std = data_np.std(axis=0) + 1e-5
    data_np = (data_np - mean) / std

    # 2. Coordinate data normalization (optional + Z-Score)
    if norm_coords:
        print("📏 Normalizing coordinates (Z-Score)...")
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_np = (coords_np - c_mean) / c_std
    else:
        print("🛡️ Using raw coordinates (Normalization Skipped).")

    train_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='train')
    val_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='val')

    print(f"✅ Data Split: Train={len(train_ds)} samples, Val={len(val_ds)} samples")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    meta = {
        "coords": coords_np,
        "gt_fine": gt_np,
        "gt_coarse": None, 
        "patch_ids": None
    }
    
    return train_loader, val_loader, meta
