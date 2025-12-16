import os
import sys
import argparse
import numpy as np
import wandb
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError:
    raise ImportError("Tigramite not installed. Please run `pip install tigramite`.")


def plot_comparison(gt_graph, est_graph, save_path):
    """Plot side-by-side heatmaps of Ground Truth vs Estimated Graph."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(gt_graph, ax=axes[0], cmap="Blues", cbar=False, square=True)
    axes[0].set_title("Ground Truth")
    
    sns.heatmap(est_graph, ax=axes[1], cmap="Reds", cbar=False, square=True)
    axes[1].set_title("PCMCI Estimated")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_continuous_data(loader):
    """Extract continuous time-series (T, N) from loader."""
    if hasattr(loader.dataset, 'data'):
        data = loader.dataset.data
        if hasattr(data, 'cpu'): data = data.cpu().numpy()
        return data

    print("Warning: Extracting data from loader iterations (Ensure shuffle=False).")
    data_list = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            data_list.append(x_np[:, -1, :]) 
        else:
            data_list.append(x_np)
    return np.concatenate(data_list, axis=0)


def run_pcmci(data, args):
    """Execute PCMCI."""
    T, N = data.shape
    print(f"Starting PCMCI on data: {data.shape}")
    
    dataframe = pp.DataFrame(data, var_names=[f"X{i}" for i in range(N)])
    
    if args.ci_test == 'parcorr':
        cond_ind_test = ParCorr(significance='analytic')
    elif args.ci_test == 'gpdpca':
        cond_ind_test = GPDPCA()
    else:
        raise ValueError(f"Unknown CI test: {args.ci_test}")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)
    results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=args.pc_alpha)
    
    p_matrix = results['p_matrix']
    est_graph = np.zeros((N, N))
    start_lag = 0 if args.include_instantaneous else 1
    
    for i in range(N):
        for j in range(N):
            if i == j: continue 
            # If significant link exists at any lag (except 0 optionally), mark edge
            if np.any(p_matrix[i, j, start_lag:] < args.pc_alpha):
                est_graph[i, j] = 1.0
                
    return est_graph, results


def main(args):
    # 1. Setup Directories
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = "PCMCI" + timestamp
    save_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. WandB Init
    run_name = f"PCMCI-{args.dataset}-{timestamp}"
    wandb.init(project=args.project_name, name=run_name, config=args, 
               mode="online" if not args.no_log else "disabled")
    
    print(f"🚀 Experiment: {args.dataset} | Out: {save_dir}")

    # 3. Data & Inference
    train_loader, _, meta = get_data_context(args)
    data = extract_continuous_data(train_loader)
    
    est_graph, _ = run_pcmci(data, args)
    
    # 4. Evaluation & Saving
    metrics = {}
    if meta.get('gt_fine') is not None:
        gt_graph = meta['gt_fine']
        
        # Align dimensions if needed
        if gt_graph.shape != est_graph.shape:
            min_dim = min(gt_graph.shape[0], est_graph.shape[0])
            est_graph = est_graph[:min_dim, :min_dim]
            gt_graph = gt_graph[:min_dim, :min_dim]

        # Calc Metrics
        metrics = count_accuracy(gt_graph, est_graph)
        print(f"\nRESULTS: F1={metrics['F1']:.4f}, SHD={metrics['SHD']}, AUROC={metrics['AUROC']:.4f}")
        
        # Plot Heatmap
        heatmap_path = os.path.join(save_dir, "heatmap.png")
        plot_comparison(gt_graph, est_graph, heatmap_path)
        metrics['Heatmap'] = wandb.Image(heatmap_path)
        
        wandb.log(metrics)
    else:
        print("No Ground Truth. Skipping evaluation.")

    # 5. Save Files (Graph & Metrics)
    np.save(os.path.join(save_dir, "est_graph.npy"), est_graph)
    
    # Save metrics to JSON for local analysis
    metrics_path = os.path.join(save_dir, "metrics.json")
    # Remove WandB Image object before JSON serialization
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
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    
    # PCMCI Config
    parser.add_argument("--ci_test", type=str, default="parcorr", choices=["parcorr", "gpdpca"])
    parser.add_argument("--tau_max", type=int, default=3)
    parser.add_argument("--pc_alpha", type=float, default=0.05)
    parser.add_argument("--include_instantaneous", action="store_true")
    
    # System Config
    parser.add_argument("--project_name", type=str, default="GeoDCD-Baselines")
    parser.add_argument("--output_dir", type=str, default="./results/baselines")
    parser.add_argument("--no_log", action="store_true")
    # Added for compatibility with some dataloaders
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--norm_coords", action="store_true")

    args = parser.parse_args()
    main(args)