import os
import argparse
import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from accelerate import Accelerator
from tqdm.auto import tqdm

from model import GeoDCD 
from dataloader import get_data_context, load_from_disk, CausalTimeSeriesDataset
from visualize import create_dynamic_gif, save_dynamic_frames_pdf
from metrics import count_accuracy

def compute_dynamic_strengths(model, x, device):
    model.eval()

    x_in = x.clone().detach().to(device).requires_grad_(True)

    results = model(x_in)

    pred_last_step = results[0]['x_pred'][:, :, -1]

    num_nodes = pred_last_step.shape[1]
    jacobian_cols = []

    for j in range(num_nodes):
        target_scalar = pred_last_step[:, j].sum()

        grads = torch.autograd.grad(
            outputs=target_scalar,
            inputs=x_in,
            retain_graph=True if j < num_nodes - 1 else False,
            create_graph=False
        )[0]

        jacobian_cols.append(grads)

    jacobian = torch.stack(jacobian_cols, dim=2)

    dynamic_adj = jacobian.abs().squeeze(0).detach().cpu()

    return dynamic_adj

def evaluate_static_graphs(model, meta, output_dir, metrics=None):
    est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()

    patch_ids = np.zeros(est_fine.shape[0])
    if hasattr(model, 'structure_S_0'):
        S_matrix = model.structure_S_0.detach().cpu().numpy()
        if S_matrix.sum() > 0:
            patch_ids = S_matrix.argmax(axis=1)

    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)

    gt_fine = meta.get('gt_fine')
    if gt_fine is None:
        gt_fine = np.zeros_like(est_fine)
    if gt_fine.ndim == 3:
        gt_fine = np.max(gt_fine, axis=0)

    coords = meta['coords']

    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab20', s=20)
    ax1.set_title(f"Swin Windows (K={len(np.unique(patch_ids))})")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_fine, cmap='Blues', vmin=0)
    ax2.set_title("GT Fine")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(est_fine, cmap='Reds', vmin=0)
    ax3.set_title("Est Fine (Soft)")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(est_fine_hard, cmap='Greens', vmin=0)
    ax4.set_title(f"Est Fine (Hard > {thresh:.2f})")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "result_static.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def run_static_analysis(model, args, accelerator, meta):
    if not accelerator.is_main_process:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
    gt_fine = meta.get('gt_fine')

    metrics = {}
    if gt_fine is not None:
        if gt_fine.ndim == 3:
            gt_fine = np.max(gt_fine, axis=0)
        metrics = count_accuracy(gt_fine, est_fine)
        accelerator.print("Static Metrics:", json.dumps(metrics, indent=4))

        if wandb.run is not None:
            wandb.log(metrics)

        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

    plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)

    if wandb.run is not None:
        wandb.log({"static_result": wandb.Image(plot_path)})
    accelerator.print(f"Static analysis completed. Saved to {plot_path}")

def run_dynamic_inference(model, args, accelerator):
    if not accelerator.is_main_process:
        return

    accelerator.print(f"\n🚀 Computing Optimized Sparse Dynamic Graph (N={args.N})...")
    try:
        base_path = getattr(args, 'data_path', 'data/synthetic')
        data_np, _, _ = load_from_disk(base_path, args.dataset, args.replica_id)

        mean = data_np.mean(axis=0)
        std = data_np.std(axis=0) + 1e-5
        data_np = (data_np - mean) / std

        full_ds = CausalTimeSeriesDataset(
            data_np, args.window_size, stride=args.window_size - 1, mode='train', split_ratio=1.0
        )
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)

        sparse_indices = []
        sparse_values = []
        max_frames = args.num_frames
        frames_generated = 0

        pbar = tqdm(total=max_frames, desc="Sparse Inference", unit="frames")

        for batch in full_loader:
            batch = batch.to(accelerator.device) # (1, N, T)

            with torch.no_grad():
                # inds: (1, N, K), vals: (1, N, K, T)
                inds, vals = model.forward_dynamic_sparse(batch)

            sparse_indices.append(inds.cpu().numpy().astype(np.int32))
            sparse_values.append(vals.cpu().numpy().astype(np.float16))

            pbar.update(1)
            frames_generated += 1

            if frames_generated >= max_frames:
                break

        pbar.close()

        if not sparse_indices:
            accelerator.print("No data generated.")
            return

        # Stack/Concatenate results
        # vals: (1, N, K, T) -> all_values: (B, N, K, T)
        # inds: (N, K) -> all_indices: (B*N, K)
        all_values = np.concatenate(sparse_values, axis=0)   # (B, N, K, T)
        all_indices = np.concatenate(sparse_indices, axis=0) # (B*N, K)
        
        # Reshape indices to (B, N, K) for consistent indexing
        all_indices = all_indices.reshape(all_values.shape[0], all_values.shape[1], -1)
        
        # Save as compressed npz
        save_path = os.path.join(args.output_dir, "est_dynamic_sparse.npz")
        np.savez_compressed(save_path, indices=all_indices, values=all_values)
        
        accelerator.print(f"✅ Sparse Dynamic Graph Saved: {save_path}")
        accelerator.print(f"   Shape: Indices {all_indices.shape}, Values {all_values.shape}")
        accelerator.print(f"   Storage: ~{all_values.nbytes / 1024**2:.2f} MB")

        # Visualization: Sample 1000 nodes if N is large
        vis_threshold = 1000
        if args.N > vis_threshold:
            accelerator.print(f"Sampling {vis_threshold} random nodes for visualization (N={args.N})...")
            np.random.seed(42)
            node_indices = np.random.choice(args.N, vis_threshold, replace=False)
            node_indices.sort()
        else:
            node_indices = np.arange(args.N)

        accelerator.print(f"Generating visualization for {len(node_indices)} nodes...")
        
        # Convert sparse to dense mean frame for visualization
        B, N, K, T = all_values.shape
        total_T = B * T
        vis_T = min(total_T, 100)
        
        N_vis = len(node_indices)
        # Reconstruct dense subset sequence (N_vis, N_vis, vis_T)
        dense_seq = np.zeros((N_vis, N_vis, vis_T))
        
        # Create map from original node index to subset index
        orig_to_sub = {orig: sub for sub, orig in enumerate(node_indices)}
        
        frame_idx = 0
        for b in range(B):
            if frame_idx >= vis_T: break
            for t in range(T):
                if frame_idx >= vis_T: break
                for sub_i, orig_i in enumerate(node_indices):
                    # For each visualized node orig_i, find its neighbors that are also in the subset
                    neigh_indices_orig = all_indices[b, orig_i] # Now shape (K,)
                    neigh_vals = all_values[b, orig_i, :, t]    # Shape (K,)
                    
                    for k in range(K):
                        orig_j = int(neigh_indices_orig[k])
                        if orig_j in orig_to_sub:
                            sub_j = orig_to_sub[orig_j]
                            dense_seq[sub_i, sub_j, frame_idx] = neigh_vals[k]
                frame_idx += 1
        
        gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
        vis_adj = dense_seq / (dense_seq.max() + 1e-9)
        create_dynamic_gif(vis_adj, gif_path, fps=8)
        
        pdf_path = os.path.join(args.output_dir, "causal_evolution.pdf")
        save_dynamic_frames_pdf(vis_adj, pdf_path)
        
        if wandb.run is not None:
            wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=8, format="gif")})
        accelerator.print(f"Visualization saved to {gif_path}")

    except Exception as e:
        accelerator.print(f"Dynamic inference failed: {e}")
        import traceback
        traceback.print_exc()

def run_full_evaluation(model, args, accelerator, meta):
    accelerator.print("\nStarting Full Evaluation...")

    model.eval()
    if accelerator.is_main_process:
        accelerator.print("Running warmup forward pass...")
        with torch.no_grad():
            dummy_x = torch.zeros(1, args.N, args.window_size).to(accelerator.device)
            model(dummy_x)

    run_static_analysis(model, args, accelerator, meta)

    accelerator.wait_for_everyone()

    if args.dynamic:
        run_dynamic_inference(model, args, accelerator)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference for GeoDCD")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.pth")
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=None, help="Number of nodes (Auto-detected if None)")
    parser.add_argument("--norm_coords", action="store_true")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[])
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Path construction: results/Dataset/Time/inference
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.dataset, timestamp, "inference")

    accelerator = Accelerator()
    accelerator.print(f"Starting Standalone Inference")
    accelerator.print(f"Output Dir: {args.output_dir}")
    accelerator.print(f"Loading model from: {args.model_path}")

    train_loader, _, meta = get_data_context(args)

    if args.N is None:
        sample_data = train_loader.dataset[0]
        if isinstance(sample_data, (list, tuple)):
            args.N = sample_data[0].shape[0]
        else:
            args.N = sample_data.shape[0]
        accelerator.print(f"Auto-detected N={args.N} from dataset")

    model = GeoDCD(
        N=args.N,
        coords=meta['coords'],
        hierarchy=args.hierarchy,
        d_model=args.d_model,
        num_bases=args.num_bases
    )

    try:
        state_dict = torch.load(args.model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        accelerator.print("Weights loaded (with strict=False).")
        if len(unexpected) > 0:
            accelerator.print(f"Ignored keys: {unexpected}")

    except Exception as e:
        accelerator.print(f"Failed to load weights: {e}")
        exit(1)

    model.to(accelerator.device)

    os.makedirs(args.output_dir, exist_ok=True)

    run_full_evaluation(model, args, accelerator, meta)

    accelerator.print(f"Inference finished. Results in {args.output_dir}")
