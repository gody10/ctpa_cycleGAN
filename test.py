"""Test script for CycleGAN with Coltea CT dataset.

This script tests a trained CycleGAN model for arterial-to-native CT translation.
It computes PSNR/SSIM metrics, saves comparison images (Input | Generated | GT), 
and exports NIfTI volumes.

Example:
    python test.py --dataroot ../data/Coltea_Processed_Nifti --name coltea_cyclegan --model cycle_gan \
        --input_nc 1 --output_nc 1 --epoch best
"""

import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from options.test_options import TestOptions
from models import create_model

# Import from your existing dataset.py
from dataset import ColteaPairedDataset, get_transforms


# --- CONFIG (can be overridden by command line args) ---
DEFAULT_TEST_CSV = "../data/Coltea-Lung-CT-100W/test_data.csv"
DEFAULT_TEST_COL = "test"
DEFAULT_DATA_ROOT = "../data/Coltea_Processed_Nifti"


def adapt_batch_for_cyclegan(batch, target_channels=1):
    """Adapt batch from ColteaPairedDataset format to CycleGAN format."""
    image = batch["image"]  # [B, C, H, W, D] from MONAI (3D)
    label = batch["label"]
    
    # Handle 3D volumes: extract 2D slices for CycleGAN (designed for 2D)
    if image.dim() == 5:
        mid_slice = image.shape[-1] // 2
        image = image[..., mid_slice]  # [B, C, H, W]
        label = label[..., mid_slice]
    
    # Ensure correct number of channels
    current_channels = image.shape[1]
    
    if target_channels == 3 and current_channels == 1:
        image = image.repeat(1, 3, 1, 1)
        label = label.repeat(1, 3, 1, 1)
    elif target_channels == 1 and current_channels == 3:
        image = image[:, 0:1, :, :]
        label = label[:, 0:1, :, :]
    
    # Normalize to [-1, 1] range expected by CycleGAN
    if image.min() >= 0:
        image = image * 2 - 1
        label = label * 2 - 1
    
    return {
        "A": image,
        "B": label,
        "A_paths": "arterial",
        "B_paths": "native"
    }


def tensor_to_numpy(tensor):
    """Convert tensor from [-1,1] to numpy array in [0,1] range."""
    img = tensor.squeeze().cpu().numpy()
    img = (img + 1) / 2.0  # [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    return img


def save_comparison_grid(art_slice, pred_slice, gt_slice, patient_id, slice_idx, output_dir):
    """
    Saves a PNG showing Input, Prediction, and Ground Truth side by side.
    
    Args:
        art_slice: 2D numpy array [H, W] (Input/Arterial)
        pred_slice: 2D numpy array [H, W] (Generated Native)
        gt_slice: 2D numpy array [H, W] (Ground Truth Native)
        patient_id: str
        slice_idx: int, the z-index of this slice
        output_dir: str
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common display settings
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    
    # 1. Input (Arterial)
    axes[0].imshow(art_slice, **kwargs)
    axes[0].set_title("Input (Arterial / Contrast)")
    axes[0].axis("off")
    
    # 2. Prediction (Generated Native)
    axes[1].imshow(pred_slice, **kwargs)
    axes[1].set_title("Generated (Native)")
    axes[1].axis("off")
    
    # 3. Ground Truth (Native)
    axes[2].imshow(gt_slice, **kwargs)
    axes[2].set_title("Ground Truth (Native)")
    axes[2].axis("off")
    
    plt.suptitle(f"{patient_id} - Slice {slice_idx}", fontsize=14)
    plt.tight_layout()
    
    # Create 'slices' subdirectory to keep things organized
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    save_path = os.path.join(slices_dir, f"{patient_id}_slice_{slice_idx:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def test_full_volume(model, batch, opt, patient_id, output_dir):
    """
    Process entire 3D volume slice by slice and save slice examples.
    """
    image = batch["image"]  # [B, C, H, W, D]
    label = batch["label"]
    
    if image.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, C, H, W, D], got {image.dim()}D")
    
    B, C, H, W, D = image.shape
    generated_slices = []
    
    # Determine indices to save (e.g., 25%, 50%, 75% of the depth)
    save_indices = {
        int(D * 0.25), 
        int(D * 0.50), 
        int(D * 0.75)
    }

    # Process each slice
    for d in range(D):
        # Extract single slice
        img_slice = image[..., d]  # [B, C, H, W]
        lbl_slice = label[..., d]
        
        # Adapt for CycleGAN
        if opt.input_nc == 3 and C == 1:
            img_slice = img_slice.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1]
        if img_slice.min() >= 0:
            img_slice = img_slice * 2 - 1
        
        # Create batch
        slice_data = {
            "A": img_slice,
            "B": lbl_slice * 2 - 1 if lbl_slice.min() >= 0 else lbl_slice,
            "A_paths": "arterial",
            "B_paths": "native"
        }
        
        # Run inference
        model.set_input(slice_data)
        model.test()
        
        # Get generated image (fake_B is A->B translation)
        visuals = model.get_current_visuals()
        if 'fake_B' in visuals:
            gen_slice = visuals['fake_B']
        elif 'fake' in visuals:
            gen_slice = visuals['fake']
        else:
            # Fallback
            for key, val in visuals.items():
                if 'real' not in key.lower():
                    gen_slice = val
                    break
        
        gen_np = tensor_to_numpy(gen_slice)
        generated_slices.append(gen_np)
        
        # --- SAVE SLICE EXAMPLES ---
        if d in save_indices:
            input_np = tensor_to_numpy(slice_data["A"])
            gt_np = tensor_to_numpy(slice_data["B"])
            
            save_comparison_grid(
                input_np, 
                gen_np, 
                gt_np, 
                patient_id, 
                d, 
                output_dir
            )
    
    # Stack slices into volume
    generated_volume = np.stack(generated_slices, axis=-1)  # [H, W, D]
    
    # Get input and ground truth volumes
    input_volume = image.squeeze().cpu().numpy()
    gt_volume = label.squeeze().cpu().numpy()
    
    # Handle channel dimension if present
    if input_volume.ndim == 4:
        input_volume = input_volume[0]
        gt_volume = gt_volume[0]
    
    return generated_volume, gt_volume, input_volume


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    
    # Get config from options or use defaults
    test_csv = getattr(opt, 'csv_path', '') or DEFAULT_TEST_CSV
    test_col = getattr(opt, 'csv_column', '') or DEFAULT_TEST_COL
    data_root = getattr(opt, 'data_root', '') or opt.dataroot or DEFAULT_DATA_ROOT
    
    # Create output directory
    output_dir = os.path.join(opt.results_dir, opt.name, f"test_{opt.epoch}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"CycleGAN Testing - {opt.name}")
    print(f"=" * 60)
    print(f"Loading model from epoch: {opt.epoch}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {opt.device}")
    
    # Create dataset using your existing dataset.py
    test_ds = ColteaPairedDataset(test_csv, test_col, data_root, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Number of test samples: {len(test_ds)}")
    
    # Create and load model
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    
    # Results storage
    results = []
    
    print(f"\nStarting inference on {len(test_ds)} patients...")
    print("-" * 40)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            patient_id = f"patient_{i:03d}"
            
            try:
                # Process full 3D volume and save examples inside
                generated_vol, gt_vol, input_vol = test_full_volume(model, batch, opt, patient_id, output_dir)
                
                # Compute metrics on full volume
                psnr_val = psnr(gt_vol, generated_vol, data_range=1.0)
                ssim_val = ssim(gt_vol, generated_vol, data_range=1.0)
                
                results.append({
                    "patient_id": patient_id,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val
                })
                
                # Save NIfTI volumes
                nib.save(
                    nib.Nifti1Image(generated_vol, affine=np.eye(4)),
                    os.path.join(output_dir, f"{patient_id}_pred.nii.gz")
                )
                
            except Exception as e:
                print(f"\nError processing {patient_id}: {e}")
                results.append({
                    "patient_id": patient_id,
                    "PSNR": float('nan'),
                    "SSIM": float('nan')
                })
                continue
    
    # Save metrics to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({len(test_ds)} Patients)")
    print("=" * 60)
    print(f"Average PSNR: {df['PSNR'].mean():.4f} ± {df['PSNR'].std():.4f}")
    print(f"Average SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  > Metrics: metrics.csv")
    print(f"  > Images:  slices/ directory")
    print(f"  > Volumes: .nii.gz files")
    print("=" * 60)