# CycleGAN Baseline for CTPA Image Translation

> **Purpose**: Fair, unmodified 2D CycleGAN baseline for comparison against the novel 3D Latent Diffusion model (`ctpa_medvae_latent_diffusion`).
>
> **Philosophy**: Do not optimize CycleGAN to win. If it produces striping artifacts or 3D inconsistencies because it processes data slice-by-slice, let it happen. That is exactly the limitation being demonstrated.

---

## Quick Start — Pick What You Need

All commands are run via Kubernetes. Edit the `args` line in `job.yml`, then submit with `kubectl create -f job.yml`.

### Option A: Retrain from scratch

```yaml
args: ["-c", "python train_with_validation_checkpoints.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_retrain --model cycle_gan --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --batch_size 4 --use_validation --patience 30 --no_html"]
```

### Option B: Resume training from a checkpoint

```yaml
args: ["-c", "python train_with_validation_checkpoints.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_with_val --model cycle_gan --input_nc 1 --output_nc 1 --continue_train --epoch_count 150 --n_epochs 100 --n_epochs_decay 100 --batch_size 4 --use_validation --patience 30 --no_html"]
```

### Option C: Test only (PSNR/SSIM metrics + visual grids)

```yaml
args: ["-c", "python test.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_with_val --model cycle_gan --input_nc 1 --output_nc 1 --epoch latest --eval"]
```

Output: `results/coltea_cyclegan_with_val/test_latest/` — per-patient NIfTI predictions, PNG comparison grids, `metrics.csv`.

### Option D: Inference + 3D stitching (for model comparison)

```yaml
args: ["-c", "python inference_and_stitch.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_with_val --model cycle_gan --input_nc 1 --output_nc 1 --epoch latest --eval"]
```

Output: `results/coltea_cyclegan_with_val/stitched_3d_epoch_latest/` — raw stitched NIfTI volumes (no post-processing).

### Option E: Compare CycleGAN vs Diffusion

```yaml
args: ["-c", "python compare_models.py --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/ --cyclegan_dir ./results/coltea_cyclegan_with_val/stitched_3d_epoch_latest/ --gt_from_cyclegan --compute_fid --output_dir ./comparison_results/"]
```

Output: `comparison_results/` — `per_patient_metrics.csv`, `summary.csv`, `summary.json`.

### Option F: Plot training loss curves

```yaml
args: ["-c", "python plot_losses.py --name coltea_cyclegan_with_val"]
```

### Using custom CSV splits (e.g. quality-filtered data)

All scripts accept `--train_csv`, `--val_csv`, `--test_csv`, and `--csv_column` flags. Defaults point to the original splits in `../data/Coltea-Lung-CT-100W/`. To use the clean (quality-filtered) CSVs instead, append:

```
--train_csv ../data/Coltea-Lung-CT-100W/clean_train_data.csv
--val_csv   ../data/Coltea-Lung-CT-100W/clean_eval_data.csv
--test_csv  ../data/Coltea-Lung-CT-100W/clean_test_data.csv
```

Available CSV files:

| File | Description |
|------|-------------|
| `train_data.csv` | Original training split (72 patients) |
| `eval_data.csv` | Original validation split (16 patients) |
| `test_data.csv` | Original test split (16 patients) |
| `clean_train_data.csv` | Quality-filtered training split |
| `clean_eval_data.csv` | Quality-filtered validation split |
| `clean_test_data.csv` | Quality-filtered test split |
| `rejected_patients.csv` | Patients excluded by quality filter |

### Submitting a Job

```bash
# Edit the args line in job.yml to one of the options above, then:
kubectl create -f job.yml
kubectl get pods                  # Check status
kubectl logs <pod-name> -f        # Stream logs
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Parity](#2-data-parity)
3. [Architecture](#3-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Dataset Classes](#5-dataset-classes)
6. [Training](#6-training)
7. [Inference & 3D Stitching](#7-inference--3d-stitching)
8. [Model Comparison (Evaluation)](#8-model-comparison-evaluation)
9. [Kubernetes Job Deployment](#9-kubernetes-job-deployment)
10. [Key File Reference](#10-key-file-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Completed Training Runs](#12-completed-training-runs)

---

## 1. Project Overview

This is a fork of the [original CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) adapted for medical CT image translation:

- **Task**: Arterial (contrast) CT → Native (non-contrast) CT
- **Data**: Coltea-Lung-CT-100W dataset (paired NIfTI volumes)
- **Training**: 2D axial slices, extracted from 3D volumes
- **Inference**: Slice-by-slice processing, stitched back into 3D — **no post-processing**

The CycleGAN serves as a **classical 2D baseline** to highlight the advantages of 3D latent diffusion for volumetric medical image translation.

---

## 2. Data Parity

Both models see **exactly the same data distribution**. This is the most critical aspect of the baseline.

### Normalization Contract

| Parameter              | CycleGAN (`dataset.py`)      | Diffusion (`src/dataset.py`) |
|------------------------|------------------------------|------------------------------|
| HU window min          | **-1000**                    | **-1000**                    |
| HU window max          | **+1000**                    | **+1000**                    |
| Window width / level   | 2000 / 0                     | 2000 / 0                     |
| Intermediate range     | [0, 1]                       | [0, 1]                       |
| Model input range      | **[-1, 1]** (Tanh)           | **[-1, 1]** (MedVAE)        |
| XY resolution          | 256 × 256                    | 256 × 256                    |
| Orientation            | RAS                          | RAS                          |
| Intensity transform    | `ScaleIntensityRanged` (MONAI) | `ScaleIntensityRanged` (MONAI) |

### What is NOT added to CycleGAN

- ❌ Body-mask-biased cropping (diffusion uses it, CycleGAN does not — this is intentional)
- ❌ Elastic deformations
- ❌ Flipping or rotation augmentations
- ❌ Any data augmentation not present in the diffusion model

### Data Directories

```
../data/Coltea_Processed_Nifti_Registered/   ← Registered NIfTI volumes (recommended)
../data/Coltea_Processed_Nifti/              ← Original unregistered
../data/Coltea-Lung-CT-100W/                 ← CSV split files
    ├── train_data.csv                       ← Training patient IDs
    ├── eval_data.csv                        ← Validation patient IDs
    └── test_data.csv                        ← Test patient IDs
```

Each patient folder contains:
```
{patient_id}/
├── arterial.nii.gz    ← Source (contrast CT)
└── native.nii.gz      ← Target (non-contrast CT)
```

CSV format (column: `patient_id`):
```csv
,patient_id
0,PD24062015
1,SMI02022021
```

---

## 3. Architecture

This is a **vanilla, unmodified CycleGAN**. No enhancements have been added.

### Generator: ResNet-9blocks

```
Input → ReflPad(3) → Conv(7,64) → IN → ReLU
  → Conv(3,128,stride=2) → IN → ReLU           (downsample)
  → Conv(3,256,stride=2) → IN → ReLU           (downsample)
  → 9 × ResNet blocks (256 channels)            (transform)
  → ConvTranspose(3,128,stride=2) → IN → ReLU  (upsample)
  → ConvTranspose(3,64,stride=2) → IN → ReLU   (upsample)
  → ReflPad(3) → Conv(7,out) → Tanh
```

### Discriminator: 70×70 PatchGAN

```
Input → Conv(4,64,stride=2) → LeakyReLU(0.2)
  → Conv(4,128,stride=2) → IN → LeakyReLU(0.2)
  → Conv(4,256,stride=2) → IN → LeakyReLU(0.2)
  → Conv(4,512,stride=1) → IN → LeakyReLU(0.2)
  → Conv(4,1,stride=1)
```

### Loss Functions

| Loss                | Type     | Weight |
|---------------------|----------|--------|
| GAN (G_A, G_B)     | LSGAN    | 1.0    |
| Cycle A→B→A        | L1       | λ_A = 10.0 |
| Cycle B→A→B        | L1       | λ_B = 10.0 |
| Identity G_A(B)≈B  | L1       | λ_idt × λ_B = 5.0 |
| Identity G_B(A)≈A  | L1       | λ_idt × λ_A = 5.0 |

### Optimizer

- Adam (lr=0.0002, β1=0.5, β2=0.999)
- Linear LR decay to 0 over the second half of training
- Image history buffer: pool_size=50

---

## 4. Directory Structure

```
ctpa_cycleGAN/
├── dataset.py                          ← Data loading with diffusion-matched normalization
├── train.py                            ← Training script (slice-based)
├── train_with_validation_checkpoints.py← Training with val L1/PSNR + early stopping
├── test.py                             ← Test: slice-by-slice 3D inference + metrics
├── inference_and_stitch.py             ← Honest 3D stitching (no post-processing)
├── compare_models.py                   ← Standalone eval: Diffusion vs CycleGAN
├── plot_losses.py                      ← Loss curve visualization
├── Dockerfile                          ← Container build
├── job.yml                             ← Kubernetes job spec (EIDF H100)
├── requirements.txt                    ← Python dependencies
├── CYCLEGAN_BASELINE.md                ← This file
│
├── models/
│   ├── cycle_gan_model.py              ← CycleGAN model (losses, forward, backward)
│   ├── networks.py                     ← Generator/Discriminator architectures
│   └── base_model.py                   ← Base class (save/load, DDP)
│
├── options/
│   ├── base_options.py                 ← Shared CLI options
│   ├── train_options.py                ← Training-specific options
│   └── test_options.py                 ← Test-specific options
│
├── data/                               ← Original CycleGAN data loaders (not used)
├── util/                               ← Visualization, image pool, utilities
├── checkpoints/                        ← Saved model weights
│   ├── coltea_cyclegan/                ← Previous training run
│   └── coltea_cyclegan_with_val/       ← Previous run with validation
└── results/                            ← Test/inference outputs
```

---

## 5. Dataset Classes

All defined in `dataset.py`:

### `ColteaSliceDataset` — For Training

- Pre-loads all NIfTI volumes into memory at construction
- Builds a flat index over every 2D axial slice across all patients
- Returns `{"A": (1,H,W), "B": (1,H,W), "A_paths": str, "B_paths": str}` in [-1, 1]
- Every slice is seen during training (not just the middle slice)

### `ColteaPairedDataset3D` — For Inference

- Returns full 3D volumes: `{"source": (1,H,W,D), "target": (1,H,W,D), "patient_id": str}` in [0, 1]
- Used by `inference_and_stitch.py` and `test.py`
- Normalization to [-1, 1] is done per-slice at inference time

### `ColteaPairedDataset` + `get_transforms()` — Legacy

- Kept for backward compatibility with older scripts
- Now uses the same fixed HU windowing (no longer percentile-based)

### Preprocessing Pipeline (shared)

```
LoadNIfTI → EnsureChannelFirst → Reorient(RAS) → ScaleIntensity[-1000,1000]→[0,1]
  → Resize(256,256,keepZ) → Pad(minZ=64) → DivisiblePad(k=16)
```

---

## 6. Training

### Basic Training

```bash
python train.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline \
    --model cycle_gan \
    --input_nc 1 --output_nc 1 \
    --n_epochs 100 --n_epochs_decay 100 \
    --batch_size 4 \
    --save_epoch_freq 5 \
    --no_html
```

### Training with Validation + Early Stopping

```bash
python train_with_validation_checkpoints.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline \
    --model cycle_gan \
    --input_nc 1 --output_nc 1 \
    --n_epochs 100 --n_epochs_decay 100 \
    --batch_size 4 \
    --use_validation --patience 30 \
    --no_html
```

### Resume from Checkpoint

```bash
python train.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline \
    --model cycle_gan \
    --input_nc 1 --output_nc 1 \
    --continue_train --epoch_count 50 \
    --n_epochs 100 --n_epochs_decay 100 \
    --no_html
```

### Key Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--train_csv` | Path to training split CSV | `../data/Coltea-Lung-CT-100W/train_data.csv` |
| `--val_csv` | Path to validation split CSV | `../data/Coltea-Lung-CT-100W/eval_data.csv` |
| `--test_csv` | Path to test split CSV | `../data/Coltea-Lung-CT-100W/test_data.csv` |
| `--csv_column` | Column name for patient IDs | `patient_id` |
| `--input_nc` | Input channels (1 for grayscale CT) | 3 |
| `--output_nc` | Output channels | 3 |
| `--netG` | Generator architecture | `resnet_9blocks` |
| `--netD` | Discriminator architecture | `basic` (PatchGAN) |
| `--ngf` | Generator base filters | 64 |
| `--ndf` | Discriminator base filters | 64 |
| `--gan_mode` | GAN loss type | `lsgan` |
| `--n_epochs` | Constant LR epochs | 100 |
| `--n_epochs_decay` | Linear decay epochs | 100 |
| `--lr` | Learning rate | 0.0002 |
| `--batch_size` | Batch size (slices) | 1 |
| `--lambda_A` | Cycle loss weight (A→B→A) | 10.0 |
| `--lambda_B` | Cycle loss weight (B→A→B) | 10.0 |
| `--lambda_identity` | Identity loss scale | 0.5 |
| `--pool_size` | Image buffer size | 50 |
| `--save_epoch_freq` | Save every N epochs | 5 |
| `--use_validation` | Enable validation loop | off |
| `--patience` | Early stopping patience | 50 |
| `--no_html` | Disable HTML visualizer | off |

### Checkpoints

Saved to `./checkpoints/{name}/`:
```
{epoch}_net_G_A.pth    ← Generator A→B
{epoch}_net_G_B.pth    ← Generator B→A
{epoch}_net_D_A.pth    ← Discriminator A
{epoch}_net_D_B.pth    ← Discriminator B
best_net_*.pth         ← Best validation model (if --use_validation)
latest_net_*.pth       ← Most recent
```

---

## 7. Inference & 3D Stitching

### Standard Test (with metrics)

```bash
python test.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline \
    --model cycle_gan \
    --input_nc 1 --output_nc 1 \
    --epoch best \
    --eval
```

**Output** (`./results/{name}/test_{epoch}/`):
- `{patient_id}_pred.nii.gz` — Generated volume
- `{patient_id}_ground_truth.nii.gz` — GT volume
- `slices/` — PNG comparison grids at 25%, 50%, 75% depth
- `metrics.csv` — Per-patient PSNR and SSIM

### Honest 3D Stitching (for model comparison)

```bash
python inference_and_stitch.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline \
    --model cycle_gan \
    --input_nc 1 --output_nc 1 \
    --epoch best \
    --eval
```

**Output** (`./results/{name}/stitched_3d_epoch_{epoch}/`):
- `{patient_id}_cyclegan_pred.nii.gz` — Raw stitched prediction
- `{patient_id}_ground_truth.nii.gz` — Ground truth
- `{patient_id}_source.nii.gz` — Source (arterial)
- `inference_manifest.csv` — Status log

> **Constraint**: No Gaussian blur, no 3D smoothing, no Z-consistency filtering. The raw slice-by-slice output is preserved as-is.

---

## 8. Model Comparison (Evaluation)

`compare_models.py` is a **standalone script** — not called during training or inference. Run it manually after both models have produced output volumes.

### Basic Usage (PSNR + SSIM)

```bash
python compare_models.py \
    --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/ \
    --cyclegan_dir ./results/coltea_cyclegan_baseline/stitched_3d_epoch_best/ \
    --gt_from_cyclegan \
    --output_dir ./comparison_results/
```

### With FID

```bash
python compare_models.py \
    --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/ \
    --cyclegan_dir ./results/coltea_cyclegan_baseline/stitched_3d_epoch_best/ \
    --gt_from_cyclegan \
    --compute_fid \
    --output_dir ./comparison_results/
```

### With Separate Ground Truth Folder

```bash
python compare_models.py \
    --diffusion_dir ../ctpa_medvae_latent_diffusion/predictions_medvae_ldm/ \
    --cyclegan_dir ./results/coltea_cyclegan_baseline/stitched_3d_epoch_best/ \
    --gt_dir /path/to/gt_volumes/ \
    --output_dir ./comparison_results/
```

### Output Files

```
comparison_results/
├── per_patient_metrics.csv   ← Per-patient PSNR/SSIM for both models
├── summary.csv               ← Mean ± std table
└── summary.json              ← Machine-readable results
```

### Metrics

| Metric | Scale | Description |
|--------|-------|-------------|
| **PSNR** | [0, 1] volumes | Peak Signal-to-Noise Ratio (higher is better) |
| **SSIM** | [0, 1] volumes | Structural Similarity, computed per-axial-slice then averaged (higher is better) |
| **FID** | 2D slices → Inception features | Fréchet Inception Distance (lower is better, optional) |

> **Scale guarantee**: Both model outputs are converted to [0, 1] before metric computation. If a model outputs [-1, 1], it is auto-detected and converted.

---

## 9. Kubernetes Job Deployment

### Build & Push Docker Image

```bash
docker build -t gody10/cycle:latest .
docker push gody10/cycle:latest
```

### Submit Training Job

Edit `job.yml` to set the desired command:

```yaml
args: ["-c", "python train.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_baseline --model cycle_gan --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --batch_size 4 --no_html"]
```

Then:
```bash
kubectl create -f job.yml
kubectl get pods                  # Check status
kubectl logs <pod-name> -f        # Stream logs
```

### Submit Inference Job

```yaml
args: ["-c", "python inference_and_stitch.py --dataroot ../data/Coltea_Processed_Nifti_Registered --name coltea_cyclegan_baseline --model cycle_gan --input_nc 1 --output_nc 1 --epoch best --eval"]
```

### Resources

The job spec requests **1× NVIDIA H100 80GB**, 20 CPU cores, 64 GB RAM.

---

## 10. Key File Reference

| File | Purpose |
|------|---------|
| `dataset.py` | Data loading with diffusion-matched HU normalization. Contains `ColteaSliceDataset`, `ColteaPairedDataset3D`, legacy `ColteaPairedDataset`. |
| `train.py` | Training loop. Uses `ColteaSliceDataset` to train on all axial slices. |
| `train_with_validation_checkpoints.py` | Training with validation L1/PSNR tracking, best-model saving, early stopping. |
| `test.py` | Slice-by-slice 3D inference with PSNR/SSIM metrics and visual grids. |
| `inference_and_stitch.py` | Honest 3D reconstruction — no post-processing. Outputs NIfTI volumes for comparison. |
| `compare_models.py` | Standalone evaluation: compares Diffusion vs CycleGAN on PSNR, SSIM, FID. |
| `plot_losses.py` | Reads training logs and plots loss curves. |
| `models/cycle_gan_model.py` | CycleGAN model: forward pass, losses, optimization. |
| `models/networks.py` | ResNet generator, PatchGAN discriminator definitions. |
| `options/base_options.py` | Shared CLI argument definitions. |

---

## 11. Troubleshooting

### "No valid patients found"
- Verify the CSV column name is `patient_id` (not `train` or `evaluation`)
- Check that `--dataroot` points to a directory containing `{patient_id}/arterial.nii.gz`

### Out of memory
- The `ColteaSliceDataset` pre-loads all volumes into RAM. For many patients, this can use significant memory.
- Reduce patient count with the `max_patients` parameter: modify the `ColteaSliceDataset(...)` call in `train.py`
- Reduce `--batch_size` for GPU OOM

### "Expected 1 input channels, got 3"
- Always pass `--input_nc 1 --output_nc 1` for grayscale CT

### Resuming with different data root
- Pass `--continue_train` and set `--dataroot` to the new path
- Ensure `--name` matches the existing checkpoint folder name

### Metrics don't match between test.py and compare_models.py
- `test.py` computes metrics on-the-fly during inference
- `compare_models.py` loads saved NIfTI volumes from disk
- Minor floating-point differences from NIfTI serialization are expected

---

## 12. Completed Training Runs

### `coltea_cyclegan_with_val` (recommended)

- **Completed**: 2026-01-20, all 200 epochs (100 constant LR + 100 linear decay)
- **Data**: `../data/Coltea_Processed_Nifti` (unregistered), CSV splits from `../data/Coltea-Lung-CT-100W/`
- **Config**: batch_size=1, lr=0.0002, lambda_A=10, lambda_B=10, lambda_identity=0.5
- **Final train loss**: ~1.82
- **Checkpoints**: `checkpoints/coltea_cyclegan_with_val/` (every 5 epochs + `latest_net_*.pth`)
- **Log**: `checkpoints/coltea_cyclegan_with_val/train_log.txt`

### `coltea_cyclegan`

- **Completed**: 200 epochs (same schedule)
- **Data**: `../data/Coltea_Processed_Nifti` (unregistered)
- **Config**: Standard training without validation tracking
- **Checkpoints**: `checkpoints/coltea_cyclegan/`

> **Note**: Both runs used the **unregistered** data (`Coltea_Processed_Nifti`). If you want to retrain on the **registered** data (`Coltea_Processed_Nifti_Registered`), use Option A from Quick Start with a new `--name` to avoid overwriting.
