import os
import argparse
import datetime
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from model import GeoDCD
from dataloader import get_data_context
from evaluate import run_full_evaluation
from metrics import count_accuracy

def train_one_epoch(model, loader, optimizer, accelerator, args):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        optimizer.zero_grad()
        results = model(x)
        
        batch_loss = 0.0
        # Accumulate losses for each layer (Recon + Pred)
        for res in results:
            x_rec, x_pred, x_target = res['x_rec'], res['x_pred'], res['x_target']
            l_rec = F.mse_loss(x_rec, x_target)
            l_pre = F.mse_loss(x_pred, x_target[..., 1:])
            batch_loss += (l_rec + l_pre)

        # Structural sparsity loss (L1)
        l1_loss = args.lambda_l1 * accelerator.unwrap_model(model).get_structural_l1_loss()
        
        final_loss = batch_loss + l1_loss
        
        accelerator.backward(final_loss)
        optimizer.step()
        total_loss += final_loss.item()
        
    return total_loss / len(loader)

def main(args):
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir)
    set_seed(args.seed)
    
    # Path and logging configuration
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        run_name = f"{args.dataset}-GeoDCD-{timestamp}"
        accelerator.init_trackers(project_name=args.project_name, config=vars(args), 
                                init_kwargs={"wandb": {"name": run_name}})
        accelerator.print(f"🚀 Experiment: {args.dataset} | Out: {args.output_dir}")

    # Data loading
    train_loader, _, meta = get_data_context(args)
    
    # Model initialization
    model = GeoDCD(
        N=args.N, 
        coords=meta['coords'], 
        hierarchy=args.hierarchy, 
        d_model=args.d_model,
        num_bases=args.num_bases
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Training loop (Unified Stage)
    progress_bar = tqdm(range(args.epochs), disable=not accelerator.is_local_main_process, desc="Training")
    
    for ep in progress_bar:
        loss = train_one_epoch(model, train_loader, optimizer, accelerator, args)
        
        if accelerator.is_main_process:
            log_dict = {"train/loss": loss}
            
            if (meta.get('gt_fine') is not None) and (meta['gt_fine'].ndim==2):
                unwrapped = accelerator.unwrap_model(model)
                est_fine = unwrapped.layers[0].graph.get_soft_graph().detach().cpu().numpy()
                metrics = count_accuracy(meta['gt_fine'], est_fine)
                log_dict.update(metrics)
                postfix_str = f"L={loss:.4f}, F1={metrics['F1']:.3f}, AUC={metrics['AUROC']:.3f}"
            else:
                postfix_str = f"L={loss:.4f}"
            
            wandb.log(log_dict)
            progress_bar.set_postfix_str(postfix_str)

    # Save and final evaluation
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save(unwrapped_model.state_dict(), save_path)
        accelerator.print(f"💾 Model saved to {save_path}")

    run_full_evaluation(unwrapped_model, args, accelerator, meta)
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--norm_coords", action="store_true")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    
    # Model
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=4)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--num_frames", type=int, default=300)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--project_name", type=str, default="GeoDCD")
    
    args = parser.parse_args()
    main(args)
