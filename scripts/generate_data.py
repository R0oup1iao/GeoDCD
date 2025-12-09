import os
import argparse
import numpy as np
from scipy.integrate import odeint
import scipy.spatial.distance as dist
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_data(data, ground_truth, coords, base_path, dataset_name, replica_id):
    """
    Saves the generated data, ground truth, and coordinates to files.
    """
    data_dir = os.path.join(base_path, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_file = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_file = os.path.join(data_dir, f'coords_{replica_id}.npy')
    
    np.save(data_file, data)
    np.save(gt_file, ground_truth)
    np.save(coords_file, coords)
    
    plt.figure(figsize=(10, 8))
    if ground_truth.ndim == 2:
        rows, cols = np.where(ground_truth != 0)
        for r, c in zip(rows, cols):
            plt.plot([coords[r, 0], coords[c, 0]], 
                     [coords[r, 1], coords[c, 1]], 
                     color='gray', alpha=0.2, linewidth=0.8, zorder=1)

    # Draw Nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=200, edgecolors='white', linewidth=1.5, zorder=2)
    
    # Annotate Node Indices
    for idx in range(coords.shape[0]):
        plt.text(coords[idx, 0], coords[idx, 1], str(idx), 
                 fontsize=9, color='white', ha='center', va='center', fontweight='bold', zorder=3)
    
    plt.title(f"Layout Preview: {dataset_name} (Replica {replica_id})\nShape: {data.shape}")
    plt.axis('equal') # Keep aspect ratio to represent geometry correctly
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save Plot
    plot_file = os.path.join(data_dir, f'layout_preview_{replica_id}.png')
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close() # Close figure to free memory
    
    logging.info(f"Saved replica {replica_id} and preview to {data_dir}")

# ==========================================
# 1. Lorenz 96 & Cluster Lorenz
# ==========================================

def generate_lorenz96_system(p, T, F=10.0, seed=0, delta_t=0.1, burn_in=1000, noise_scale=0.1, center=(0,0), radius=5.0):
    """
    Core generator for a single Lorenz96 ring.
    Returns: X (T, p), gt (p, p), coords (p, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # --- ODE Definition ---
    def lorenz96_deriv(x, t):
        x_plus_1 = np.roll(x, -1)
        x_minus_1 = np.roll(x, 1)
        x_minus_2 = np.roll(x, 2)
        dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
        return dxdt

    # --- Integration ---
    x0 = np.random.normal(scale=0.01, size=p) + F 
    total_steps = T + burn_in
    t = np.linspace(0, total_steps * delta_t, total_steps)
    
    X_full = odeint(lorenz96_deriv, x0, t)
    X = X_full[burn_in:, :]
    
    if noise_scale > 0:
        X += np.random.normal(scale=noise_scale, size=X.shape)

    # --- GT Construction (Ring Topology) ---
    # x_{i} depends on x_{i-1}, x_{i-2}, x_{i+1}
    gt = np.zeros((p, p), dtype=int)
    for i in range(p):
        gt[(i - 1) % p, i] = 1
        gt[(i - 2) % p, i] = 1
        gt[(i + 1) % p, i] = 1
        gt[i, i] = 1
        
    # --- Coordinates ---
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    # Coords centered at 'center'
    coords = np.stack([radius * np.cos(angles) + center[0], 
                       radius * np.sin(angles) + center[1]], axis=1)
    
    # Slight jitter
    coords += np.random.normal(0, 0.1, coords.shape)

    return X, gt, coords

def generate_cluster_lorenz(p, T, seed, num_groups=4):
    """
    Generates multiple independent Lorenz96 rings placed in a grid.
    If p=32 and num_groups=4, we get 4 rings of size 8.
    """
    if p % num_groups != 0:
        raise ValueError(f"p ({p}) must be divisible by num_groups ({num_groups})")
    
    nodes_per_group = p // num_groups
    
    all_data = []
    all_gt = np.zeros((p, p), dtype=int)
    all_coords = []
    
    # Grid layout for centers
    grid_size = int(np.ceil(np.sqrt(num_groups)))
    spacing = 15.0 # Distance between cluster centers
    
    for g in range(num_groups):
        # Calculate center position
        row = g // grid_size
        col = g % grid_size
        center = (col * spacing, row * spacing)
        
        # Vary seed slightly for each group
        group_seed = seed + g * 100
        
        X, gt, coords = generate_lorenz96_system(
            p=nodes_per_group, 
            T=T, 
            seed=group_seed,
            center=center,
            radius=3.0 # Slightly smaller radius for clusters
        )
        
        all_data.append(X)
        all_coords.append(coords)
        
        # Fill Block Diagonal GT
        start = g * nodes_per_group
        end = (g + 1) * nodes_per_group
        all_gt[start:end, start:end] = gt

    final_data = np.concatenate(all_data, axis=1) # (T, p)
    final_coords = np.concatenate(all_coords, axis=0) # (p, 2)
    
    return final_data, all_gt, final_coords

# ==========================================
# 2. TVSEM (Expanded)
# ==========================================

def generate_tvsem_expanded(p, T, seed):
    """
    Generates TVSEM data by creating p/2 independent pairs.
    Each pair follows the switching causality logic.
    """
    if p % 2 != 0:
        raise ValueError(f"For TVSEM, p ({p}) must be even.")
        
    np.random.seed(seed)
    
    num_pairs = p // 2
    all_data = []
    all_gt = np.zeros((T, p, p), dtype=int) # TVSEM GT is time-varying!
    all_coords = []
    
    # Grid layout for pairs
    grid_cols = int(np.ceil(np.sqrt(num_pairs)))
    spacing = 5.0

    for i in range(num_pairs):
        # Indices for this pair in the global matrix
        idx_0 = 2 * i
        idx_1 = 2 * i + 1
        
        # --- Generate Pair Data ---
        pair_data = np.zeros((T, 2))
        errors = np.random.normal(0, 0.1, (T, 2))
        pair_data[0] = np.random.normal(0, 1, 2)
        
        # Determine spatial center
        row = i // grid_cols
        col = i % grid_cols
        center_x, center_y = col * spacing, row * spacing
        
        # Random coords around center
        pair_coords = np.random.randn(2, 2) * 0.5 + np.array([center_x, center_y])
        all_coords.append(pair_coords)

        for t in range(1, T):
            segment_index = (t - 1) // 400
            
            # Independent logic per pair
            if segment_index % 2 == 0: 
                # Regime A: 1 -> 0
                at, bt = 0.8, 0.1
                # GT: index 1 causes index 0
                all_gt[t, idx_1, idx_0] = 1 
            else: 
                # Regime B: 0 -> 1
                at, bt = 0.2, 0.7
                # GT: index 0 causes index 1
                all_gt[t, idx_0, idx_1] = 1 

            pair_data[t, 0] = at * pair_data[t-1, 1] + errors[t, 0] 
            pair_data[t, 1] = bt * pair_data[t-1, 0] + errors[t, 1]
            
        all_data.append(pair_data)

    final_data = np.concatenate(all_data, axis=1) # (T, p)
    final_coords = np.concatenate(all_coords, axis=0) # (p, 2)
    
    # TVSEM usually returns Time-Varying GT (T, p, p)
    # If your pipeline expects static GT, you might need to aggregate it, 
    # but here we return the full dynamic GT.
    return final_data, all_gt, final_coords

# ==========================================
# 3. NC (Non-linear Causal) Expanded
# ==========================================

def generate_nc_block(T, seed, offset_idx):
    """
    Generates a single block of NC8 (8 variables).
    """
    np.random.seed(seed)
    N = 8
    data = np.zeros((T, N))
    gt = np.zeros((N, N), dtype=int)
    
    # Static GT Construction (Relative indices 0-7)
    gt[0, 1] = 1; gt[1, 1] = 1
    gt[0, 2] = 1
    gt[2, 3] = 1
    gt[4, 5] = 1; gt[5, 5] = 1
    gt[4, 6] = 1; gt[0, 6] = 1
    gt[0, 7] = 1; gt[4, 7] = 1
    
    # Initialization
    data[:16] = np.random.normal(0, 0.1, (16, N))
    errors = np.random.normal(0, 1, (T, N)) 
    
    # Simulation
    for t in range(16, T):
        # [FIXED] Correct slicing logic:
        # 1. Slice positively from t-16 to t (exclusive), giving indices t-16...t-1
        # 2. Reverse it with [::-1], giving t-1...t-16
        # 3. Transpose so rows are variables, cols are time lags
        hist = data[t-16 : t][::-1].T 
        
        xt, yt, zt, wt = hist[0], hist[1], hist[2], hist[3]
        at, bt, ct, ot = hist[4], hist[5], hist[6], hist[7]
        
        # Equations
        data[t, 0] = 0.45 * np.sin(t / (4 * np.pi)) + 0.45 * np.sin(t / (9 * np.pi)) + 0.25 * np.sin(t / (3 * np.pi)) + 0.1 * errors[t, 0]
        data[t, 1] = 0.24*xt[0] - 0.28*xt[1] + 0.08*xt[2] + 0.2*xt[3] + 0.2*yt[0] - 0.12*yt[1] + 0.16*yt[2] + 0.04*yt[3] + 0.02*errors[t, 1]
        data[t, 2] = 3*(0.6*xt[0])**3 + 3*(0.4*xt[1])**3 + 3*(0.2*xt[2])**3 + 3*(0.5*xt[3])**3 + 0.02*errors[t, 2]
        data[t, 3] = 0.8*(0.4*zt[0])**3 + 0.8*(0.5*zt[1])**3 + 0.64*zt[2] + 0.48*zt[3] + 0.02*errors[t, 3]
        data[t, 4] = 0.15*np.sin(t/6) + 0.35*np.sin(t/80) + 0.65*np.sin(t/125) + 0.1*errors[t, 4]
        data[t, 5] = 0.54*at[12] - 0.63*at[13] + 0.18*at[14] + 0.45*at[15] + 0.36*bt[12] + 0.27*bt[13] - 0.36*bt[14] + 0.18*bt[15] + 0.02*errors[t, 5]
        
        term_6_1 = 0.24*at[12] + 0.3*at[13]
        term_6_2 = 0.2*at[14] + 0.5*xt[15] 
        data[t, 6] = np.maximum(term_6_1, -0.2) + 1.2*np.abs(term_6_2) + 0.02*errors[t, 6]
        data[t, 7] = 0.39*xt[12] - 0.65*xt[13] + 0.52*xt[14] + 0.13*xt[15] + 0.52*at[0] - 0.65*at[1] + 0.26*at[2] + 0.52*at[3] + 0.02*errors[t, 7]

    return data, gt

def generate_nc_expanded(p, T, seed):
    """
    Generates p variables by tiling NC8 blocks.
    p must be a multiple of 8.
    """
    if p % 8 != 0:
        raise ValueError(f"For NC dataset, p ({p}) must be a multiple of 8.")
        
    num_blocks = p // 8
    all_data = []
    all_gt = np.zeros((p, p), dtype=int)
    all_coords = []
    
    # Layout
    grid_cols = int(np.ceil(np.sqrt(num_blocks)))
    spacing = 8.0
    
    for b in range(num_blocks):
        block_seed = seed + b * 50
        X_block, gt_block = generate_nc_block(T, block_seed, offset_idx=b*8)
        
        all_data.append(X_block)
        
        # GT Offset
        start = b * 8
        end = (b + 1) * 8
        all_gt[start:end, start:end] = gt_block
        
        # Coords (Cluster cloud)
        row = b // grid_cols
        col = b % grid_cols
        center_x, center_y = col * spacing, row * spacing
        
        # 8 points randomly distributed near center
        block_coords = np.random.randn(8, 2) * 1.5 + np.array([center_x, center_y])
        all_coords.append(block_coords)
        
    final_data = np.concatenate(all_data, axis=1)
    final_coords = np.concatenate(all_coords, axis=0)
    
    return final_data, all_gt, final_coords

# ==========================================
# 4. VAR with NetworkX Coordinates
# ==========================================

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def simulate_spatial_var(p, T, lag=2, neighbors=3, beta_scale=1.0, sd=0.1, seed=0):
    """
    生成基于几何距离的 VAR 模型。
    1. 先生成坐标。
    2. 每个节点只连接最近的 'neighbors' 个点。
    3. 系数大小与距离成反比。
    """
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.uniform(0, 10, size=(p, 2))
    dists = dist.cdist(coords, coords, metric='euclidean')
    
    GC = np.eye(p, dtype=int) 
    beta = np.zeros((p, p))
    np.fill_diagonal(beta, 0.4) 

    for i in range(p):
        nearest_indices = np.argsort(dists[i])[1 : neighbors + 1]
        
        for neighbor_j in nearest_indices:
            dist_val = dists[i, neighbor_j]
            weight = np.exp(-dist_val**2 / 10.0) 
            sign = np.random.choice([1, -1])
            coeff = sign * beta_scale * weight / np.sqrt(neighbors)
            
            beta[i, neighbor_j] = coeff
            GC[i, neighbor_j] = 1
    beta_full_list = [beta]
    for l in range(1, lag):
        decay_factor = 0.2 ** l
        beta_lag = np.eye(p) * 0.2 * decay_factor
        beta_full_list.append(beta_lag)

    beta_concat = np.hstack(beta_full_list)
    beta_final = make_var_stationary(beta_concat)
    burn_in = 200
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    
    for t in range(lag, T + burn_in):
        hist = X[:, (t-lag):t].flatten(order='F')
        X[:, t] = np.dot(beta_final, hist) + errors[:, t]

    return X.T[burn_in:], beta_final, GC, coords


def generate_var_system(p, T, lag=2, sparsity=None, seed=0):
    if sparsity is None:
        k = 4
    else:
        k = int(p * sparsity)
        if k < 2: k = 2
        if k > 10: k = 10
    
    X, beta, GC, coords = simulate_spatial_var(
        p, T, lag=lag, neighbors=k, beta_scale=0.9, seed=seed
    )
    
    return X, GC, coords


# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for GeoDCD.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['lorenz96', 'cluster_lorenz', 'tvsem', 'nc', 'var'],
                        help='Name of the dataset to generate.')
    parser.add_argument('--num_replicas', type=int, default=5, help='Number of replicas to generate.')
    parser.add_argument('--output_path', type=str, default='data/synthetic', help='Base path to save the data.')
    parser.add_argument('--T', type=int, default=1000, help='Time steps.')
    parser.add_argument('--p', type=int, default=32, help='Total number of variables (nodes).')
    
    parser.add_argument('--num_groups', type=int, default=4, help='Number of groups for cluster_lorenz.')
    parser.add_argument('--var_lag', type=int, default=2, help='Lag order for VAR model.')
    parser.add_argument('--var_sparsity', type=float, default=0.03, help='Sparsity for VAR coefficients.')
    
    args = parser.parse_args()

    for i in range(args.num_replicas):
        seed = 42 + i * 100 
        logging.info(f"--- Generating {args.dataset}, Replica {i+1}/{args.num_replicas} (seed={seed}) ---")

        if args.dataset == 'lorenz96':
            data, gt, coords = generate_lorenz96_system(p=args.p, T=args.T, F=10, seed=seed)
            
        elif args.dataset == 'cluster_lorenz':
            data, gt, coords = generate_cluster_lorenz(p=args.p, T=args.T, seed=seed, num_groups=args.num_groups)
            
        elif args.dataset == 'tvsem':
            data, gt, coords = generate_tvsem_expanded(p=args.p, T=args.T, seed=seed)
            
        elif args.dataset == 'nc':
            data, gt, coords = generate_nc_expanded(p=args.p, T=args.T, seed=seed)
        
        elif args.dataset == 'var':
            data, gt, coords = generate_var_system(
                p=args.p, 
                T=args.T, 
                lag=args.var_lag, 
                sparsity=args.var_sparsity, 
                seed=seed
            )
            
        else:
            raise ValueError("Unknown dataset.")
            
        save_data(data, gt, coords, args.output_path, args.dataset, i)

if __name__ == '__main__':
    main()