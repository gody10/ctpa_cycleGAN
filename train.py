"""Training script for GAN baselines (CycleGAN / pix2pix) with data-parity to 3D Latent Diffusion.

Uses ColteaSliceDataset which:
  - Loads NIfTI volumes with fixed HU windowing [-1000, 1000] (matches diffusion model)
  - Extracts ALL 2D axial slices (not just the middle slice)
  - Returns slices in [-1, 1] ready for the model

Example (CycleGAN):
    python train.py --dataroot ../data/Coltea_Processed_Nifti_Registered \
        --name coltea_cyclegan_baseline --model cycle_gan \
        --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100

Example (pix2pix):
    python train.py --dataroot ../data/Coltea_Processed_Nifti_Registered \
        --name coltea_pix2pix_baseline --model pix2pix \
        --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100
"""

import time
import logging
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

# New slice-based dataset with diffusion-matched normalization
from dataset import ColteaSliceDataset


# Defaults are now defined in options/base_options.py (--train_csv, --val_csv, --csv_column)


def setup_logging(opt):
    """Setup logging to file and console with detailed tracking."""
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train_log.txt')
    
    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Append mode for resume
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Log configuration at start
    logger.info("=" * 60)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Checkpoints dir: {opt.checkpoints_dir}/{opt.name}")
    
    # Log key training parameters
    logger.info("-" * 40)
    logger.info("Training Configuration:")
    logger.info(f"  Model: {opt.model}")
    logger.info(f"  Batch size: {opt.batch_size}")
    logger.info(f"  Learning rate: {opt.lr}")
    logger.info(f"  Epochs: {opt.n_epochs} + {opt.n_epochs_decay} decay")
    logger.info(f"  Input channels: {opt.input_nc}")
    logger.info(f"  Output channels: {opt.output_nc}")
    logger.info(f"  Generator: {opt.netG}")
    logger.info(f"  Discriminator: {opt.netD}")
    logger.info(f"  GAN mode: {opt.gan_mode}")
    logger.info(f"  Lambda A: {getattr(opt, 'lambda_A', 'N/A')}")
    logger.info(f"  Lambda B: {getattr(opt, 'lambda_B', 'N/A')}")
    logger.info(f"  Lambda identity: {getattr(opt, 'lambda_identity', 'N/A')}")
    logger.info("-" * 40)
    
    return logger




if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    
    # Setup logging
    logger = setup_logging(opt)
    
    # CSV paths and data root from CLI (defaults in base_options.py)
    train_csv = opt.train_csv
    train_col = opt.csv_column
    val_csv = opt.val_csv
    val_col = opt.csv_column
    data_root = opt.dataroot

    patience = getattr(opt, 'patience', 50)
    use_validation = getattr(opt, 'use_validation', False)
    
    # Create slice-based dataset: loads all NIfTI volumes, indexes individual slices
    # This ensures every axial slice is seen during training (not just the middle one)
    logger.info("Loading training volumes and building slice index ...")
    train_ds = ColteaSliceDataset(train_csv, train_col, data_root)
    train_loader = DataLoader(
        train_ds, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=int(opt.num_threads), 
        pin_memory=True
    )
    
    dataset_size = len(train_ds)
    logger.info(f"The number of training slices = {dataset_size}")
    
    # Create validation loader if enabled
    val_loader = None
    if use_validation and os.path.exists(val_csv):
        logger.info("Loading validation volumes ...")
        val_ds = ColteaSliceDataset(val_csv, val_col, data_root)
        val_loader = DataLoader(
            val_ds, 
            batch_size=opt.batch_size, 
            shuffle=False, 
            num_workers=max(1, int(opt.num_threads) // 2), 
            pin_memory=True
        )
        logger.info(f"The number of validation slices = {len(val_ds)}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    
    total_iters = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    start_msg = f"Starting Training on {opt.device} | Model: {opt.model}"
    logger.info(start_msg)
    if use_validation:
        logger.info(f"Validation enabled with patience={patience}")
    
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_loss = 0.0
        num_batches = 0
        
        visualizer.reset()

        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        
        for batch in progress_bar:
            iter_start_time = time.time()
            
            # ColteaSliceDataset already returns {"A", "B", "A_paths", "B_paths"}
            # in [-1, 1] with shape (B, 1, H, W) — no adaptation needed
            data = batch
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters()
            
            # Track epoch loss
            losses = model.get_current_losses()
            batch_loss = sum(losses.values())
            epoch_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, 0)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

        # Compute average training loss for epoch
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Get current learning rate
        current_lr = model.optimizers[0].param_groups[0]['lr'] if model.optimizers else opt.lr
        
        # Log detailed losses for this epoch
        epoch_losses = model.get_current_losses()
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()])
        logger.info(f"Epoch {epoch} Losses: {loss_str}")
        logger.info(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.6f}")
        
        # --- VALIDATION STEP ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    data = batch
                    model.set_input(data)
                    model.forward()
                    
                    losses = model.get_current_losses()
                    val_loss += sum(losses.values())
                    val_batches += 1
            
            model.train()
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.6f}")
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                model.save_networks('best')
                logger.info(f" -> New Best Model Saved! (Val Loss: {best_val_loss:.6f})")
            else:
                early_stopping_counter += 1
                logger.info(f" -> No improvement. Early stopping counter: {early_stopping_counter}/{patience}")
            
            # Check early stopping
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"End of epoch {epoch} / {total_epochs} \t Time Taken: {epoch_time:.0f} sec")

    # Save final model
    model.save_networks('final')
    logger.info("Training completed. Final model saved.")
    
    cleanup_ddp()
