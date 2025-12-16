import os
import sys
import argparse
import numpy as np
import wandb
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from causallearn.search.FCMBased import lingam

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

try:
    from dataloader import get_data_context
    from metrics import count_accuracy
except ImportError as e:
    print(f"Error importing modules from src: {e}")
    sys.exit(1)


def plot_comparison(gt_graph, est_graph, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(gt_graph, ax=axes[0], cmap="Blues", cbar=False, square=True)
    axes[0].set_title("Ground Truth")
    
    sns.heatmap(est_graph, ax=axes[1], cmap="RdBu", center=0, cbar=True, square=True)
    axes[1].set_title("VAR-LiNGAM Estimated (Lag-1)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_continuous_data(loader):
    if hasattr(loader.dataset, 'data'):
        data = loader.dataset.data
        if hasattr(data, 'cpu'): data = data.cpu().numpy()
        return data

    print("Warning: Concatenating data from loader (Ensure shuffle=False).")
    data_list = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            data_list.append(x_np[:, -1, :]) 
        else:
            data_list.append(x_np)
    return np.concatenate(data_list, axis=0)


def run_varlingam(data, args):
    """
    Run VAR-LiNGAM.
    Note: LiNGAM returns B where X_t = B * X_{t-1}. 
    Row is Effect, Col is Cause.
    Metrics usually expect Row=Source, Col=Target. Thus we return Transpose.
    """
    print(f"Starting VAR-LiNGAM on data: {data.shape}")
    
    # criterion=None disables auto lag search if we want fixed lags
    model = lingam.VARLiNGAM(lags=args.lags, criterion=args.criterion, prune=args.prune)
    model.fit(data)
    
    # Extract Lag-1 Matrix (index 0)
    # adjacency_matrices_ shape: (lags, N, N)
    adj_matrix = model.adjacency_matrices_[0]
    
    # Transpose to obtain [Source, Target] orientation for metrics
    est_graph = adj_matrix.T
    
    return est_graph, model


def main(args):
    # 1. Setup
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, args.dataset, f"VARLiNGAM_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    run_name = f"VARLiNGAM-{args.dataset}-{timestamp}"
    wandb.init(project=args.project_name, name=run_name, config=args, 
               mode="online" if not args.no_log else "disabled")
    
    print(f"🚀 Experiment: {args.dataset} | Out: {save_dir}")

    # 2. Data & Inference
    train_loader, _, meta = get_data_context(args)
    data = extract_continuous_data(train_loader)
    
    est_graph, _ = run_varlingam(data, args)
    
    # 3. Evaluation
    metrics = {}
    if meta.get('gt_fine') is not None:
        gt_graph = meta['gt_fine']
        
        # Align dimensions
        min_dim = min(gt_graph.shape[0], est_graph.shape[0])
        est_graph = est_graph[:min_dim, :min_dim]
        gt_graph = gt_graph[:min_dim, :min_dim]

        # Binarize for classification metrics if needed, or pass weighted
        # Assuming metrics.py handles weighted or we just care about non-zero structure
        est_binary = (np.abs(est_graph) > 0).astype(float)
        
        metrics = count_accuracy(gt_graph, est_binary)
        print(f"\nRESULTS: F1={metrics.get('F1', 0):.4f}, SHD={metrics.get('SHD', -1)}")
        
        heatmap_path = os.path.join(save_dir, "heatmap.png")
        plot_comparison(gt_graph, est_graph, heatmap_path)
        metrics['Heatmap'] = wandb.Image(heatmap_path)
        
        wandb.log(metrics)
    else:
        print("No Ground Truth. Skipping evaluation.")

    # 4. Save
    np.save(os.path.join(save_dir, "est_graph.npy"), est_graph)
    
    metrics_path = os.path.join(save_dir, "metrics.json")
    json_metrics = {k: v for k, v in metrics.items() if not isinstance(v, wandb.Image)}
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
        
    print(f"✅ Saved results to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data Config
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--N", type=int, default=128) # 默认为10便于调试，实际运行需匹配数据
    parser.add_argument("--batch_size", type=int, default=256)
    
    # VAR-LiNGAM Config
    parser.add_argument("--lags", type=int, default=1, help="Number of lags (default 1 for Lorenz96)")
    parser.add_argument("--criterion", type=str, default=None, choices=[None, 'bic', 'aic'], help="Auto lag selection criterion")
    parser.add_argument("--prune", action="store_true", default=True, help="Prune small coefficients")
    
    # System Config
    parser.add_argument("--project_name", type=str, default="GeoDCD-Baselines")
    parser.add_argument("--output_dir", type=str, default="./results/baselines")
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--norm_coords", action="store_true")

    args = parser.parse_args()
    main(args)
    
    
"""
python baselines/varlingam.py \
  --dataset lorenz96 \
  --N 128 \
  --lags 1 \
  --prune \
  --batch_size 256 \
  --project_name "GeoDCD-Baselines-varlingam"
"""