import os
import argparse
import datetime
import torch
import torch.nn.functional as F
from torch.func import jacrev, functional_call
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from accelerate import Accelerator
from tqdm.auto import tqdm

from model import GeoDCD 
from dataloader import get_data_context, load_from_disk, CausalTimeSeriesDataset
from visualize import create_dynamic_gif
from metrics import count_accuracy

def compute_dynamic_strengths(model, x, device, chunk_size=1):
    """
    Compute dynamic causal strengths
    """ 
    model.eval()
    x_in = x.detach().clone().to(device).requires_grad_(True) # (1, N, T)

    results = model(x_in)
    x_pred = results[0]['x_pred']
    x_pred = x_pred.squeeze(0) # (N, T)
    N, T = x_pred.shape
    
    jacobian_rows = []
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        target_scalar = x_pred[i:end_idx].sum() 
        retain_graph = True if end_idx < N else False
        
        grads = torch.autograd.grad(
            outputs=target_scalar, 
            inputs=x_in, 
            retain_graph=retain_graph,
            create_graph=False,
            only_inputs=True
        )[0]
        
        jacobian_rows.append(grads.squeeze(0).detach().cpu())
        del grads
        torch.cuda.empty_cache()
    jacobian = torch.stack(jacobian_rows, dim=0)
    dynamic_adj = jacobian.permute(1, 0, 2).abs() 
    return dynamic_adj[..., :-1]

def evaluate_static_graphs(model, meta, output_dir, metrics=None):
    """
    Plot static analysis graphs (Swin Windows, GT, Est Soft, Est Hard)
    """
    # 1. Extract Fine Graph (Level 0)
    est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
    
    # 2. Extract Swin Window partitions
    patch_ids = np.zeros(est_fine.shape[0])
    if hasattr(model, 'structure_S_0'):
        S_matrix = model.structure_S_0.detach().cpu().numpy() 
        if S_matrix.sum() > 0:
            patch_ids = S_matrix.argmax(axis=1)
    
    # Hard thresholding
    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)
    
    # GT check
    gt_fine = meta.get('gt_fine')
    if gt_fine is None:
        gt_fine = np.zeros_like(est_fine) 
    if gt_fine.ndim == 3:
        gt_fine = np.max(gt_fine, axis=0)
    
    coords = meta['coords']
    
    # Plotting
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

def run_full_evaluation(model, args, accelerator, meta):
    accelerator.print("\n📊 Starting Full Evaluation...")
    
    # Warmup forward pass
    model.eval()
    if accelerator.is_main_process:
        accelerator.print("⚙️  Running warmup forward pass to initialize Geometric Poolers...")
        with torch.no_grad():
            dummy_x = torch.zeros(1, args.N, args.window_size).to(accelerator.device)
            model(dummy_x)
    
    # 1. Static graph evaluation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True) 
        
        est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        if gt_fine.ndim == 3:
            gt_fine = np.max(gt_fine, axis=0)
        
        metrics = {}
        if gt_fine is not None:
            metrics = count_accuracy(gt_fine, est_fine)
            accelerator.print("Static Metrics:", json.dumps(metrics, indent=4))
            
            if wandb.run is not None: 
                wandb.log(metrics)
            
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

        plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)
        
        if wandb.run is not None:
            wandb.log({"static_result": wandb.Image(plot_path)})

    accelerator.wait_for_everyone()

    # 2. Dynamic graph inference (only main process)
    if accelerator.is_main_process:
        accelerator.print(f"\nComputing Dynamic Graph (Jacobian)...")
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
            
            all_strengths = []
            max_frames = 200  # Can be increased for smoother GIF
            frames_generated = 0
            
            # OOM prevention: Chunk Size
            current_chunk_size = 1

            # --- TQDM modification start ---
            # Based on max_frames target rather than loader length
            pbar = tqdm(total=max_frames, desc="Generating Frames", unit="frames")
            
            for batch in full_loader:
                batch = batch.to(accelerator.device)
                
                # Compute causal graph for current batch
                curr_adj = compute_dynamic_strengths(model, batch, accelerator.device, chunk_size=current_chunk_size)
                
                all_strengths.append(curr_adj.numpy()) 
                
                # Get number of frames generated
                n_new_frames = curr_adj.shape[2]
                
                # Manually update progress bar
                pbar.update(n_new_frames)
                frames_generated += n_new_frames
                
                if frames_generated >= max_frames: 
                    break
            
            pbar.close()
            # --- TQDM modification end ---
            
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)[:, :, :max_frames]
            
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            
            # GIF Visualization
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            vis_adj = final_dynamic_adj / (final_dynamic_adj.max() + 1e-9)
            create_dynamic_gif(vis_adj, gif_path, fps=8)
            
            if wandb.run is not None:
                wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=8, format="gif")})
            accelerator.print(f"✅ GIF saved to {gif_path}")
            
        except Exception as e:
            accelerator.print(f"❌ Dynamic inference failed: {e}")
            import traceback
            traceback.print_exc()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference for GeoDCD")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.pth")
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--norm_coords", action="store_true")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[32, 8])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Path construction: results/Dataset/Time/inference
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.dataset, timestamp, "inference")

    accelerator = Accelerator()
    accelerator.print(f"🚀 Starting Standalone Inference")
    accelerator.print(f"📂 Output Dir: {args.output_dir}")
    accelerator.print(f"📂 Loading model from: {args.model_path}")
    
    _, _, meta = get_data_context(args)
    
    model = GeoDCD(
        N=args.N, 
        coords=meta['coords'], 
        hierarchy=args.hierarchy, 
        d_model=args.d_model,
        num_bases=args.num_bases
    )
    
    # Load weights
    try:
        state_dict = torch.load(args.model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        accelerator.print("✅ Weights loaded (with strict=False).")
        if len(unexpected) > 0:
            accelerator.print(f"⚠️  Ignored keys: {unexpected}")
            
    except Exception as e:
        accelerator.print(f"❌ Failed to load weights: {e}")
        exit(1)
        
    model.to(accelerator.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_full_evaluation(model, args, accelerator, meta)
    
    accelerator.print(f"✨ Inference finished. Results in {args.output_dir}")

"""
python src/evaluate.py \
    --dataset SD \
    --data_path data/real \
    --model_path ./results/SD/20251128_234020/model.pth\
    --N 716 \
    --hierarchy 64 8 \
    --window_size 12
    
accelerate launch src/evaluate.py \
    --dataset GBA \
    --data_path data/real \
    --model_path ./results/GBA/20251129_135233/model.pth\
    --N 2352 \
    --hierarchy 512 64 8 \
    --window_size 12
    
python src/evaluate.py \
    --dataset GLA \
    --data_path data/real \
    --model_path ./results/GLA/20251129_234400/model.pth\
    --N 3834 \
    --hierarchy 64 8 \
    --window_size 12

"""
