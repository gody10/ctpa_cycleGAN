# Evaluation Metrics — Ranked Reference (Baseline: CycleGAN / Pix2Pix)

**Last updated:** April 2026  
**Context:** This document mirrors `ctpa_medvae_latent_diffusion/docs/evaluation_metrics_ranked.md` for the GAN baseline repository. Rankings and clinical rationale are identical; what differs is noted explicitly.

---

## The Revised Research Question

The primary goal is **not** to produce a high-quality synthetic non-contrast CT. It is to produce a **synthetic subtraction image** — (arterial − predicted_native) — that carries the same diagnostic information as the real dual-phase subtraction (arterial − real_native), such that acquiring a separate non-contrast scan becomes unnecessary for PE detection.

This reframing changes which metrics matter most. Non-contrast CT quality (PSNR, SSIM) is a supporting check, not the headline. The quality of the subtraction image in the pulmonary vasculature is the primary claim.

---

## Anatomy of the Region-of-Interest Masks

### Threshold-Based ROI Mask (`roi_*` metrics)
**Definition:** All voxels where |GT arterial − GT native| > 50 HU.  
**Covers:** Every structure that enhanced with contrast — dominated by:

| Structure | Approximate share of enhanced volume | PE relevance |
|---|---|---|
| Right heart (right atrium + right ventricle) | ~40% | Indirect (right heart strain is a PE complication, not PE itself) |
| Left heart (left atrium + left ventricle) | ~20% | None for PE detection |
| Aorta (ascending, arch, descending) | ~15% | None for PE |
| Pulmonary arteries (trunk, lobar, segmental) | ~15% | **Direct — PE lodges here** |
| Pulmonary veins | ~7% | None for PE |
| SVC / IVC / other veins | ~3% | None for PE |

**Verdict:** ~60% dominated by cardiac chambers. Clinically noisy for PE evaluation. Valid for model-vs-model comparisons within the same script, but the absolute numbers are not clinically interpretable on their own.

---

### TotalSegmentator PE Mask (`pe_roi_*` and `fair_sub_*` metrics)
**Definition:** Pre-computed TotalSegmentator segmentation of `aorta`, `superior_vena_cava`, `lung_arteries`, `lung_veins`.  
**Covers:** Named pulmonary vessels and great veins. Excludes cardiac chambers.  
**Coverage of total enhanced volume:** ~36–46%.

| Structure | PE relevance |
|---|---|
| Lung arteries (trunk, lobar, segmental) | **Direct — primary PE site** |
| Lung veins | Low — not where PE lodges |
| Aorta | None for PE |
| Superior vena cava | None for PE |

**Critical limitation:** TotalSegmentator performs well on main and lobar pulmonary arteries (Dice ~0.94–0.97) but systematically **misses subsegmental and peripheral arteries** — the vessels where PE most commonly occurs.

**Note:** The GAN baselines do not use a correction-network mask during inference, so there is no background-copy artifact. All subtraction voxels are genuinely predicted (not forced to zero outside a mask), which means `sub_psnr` and `sub_ssim` are **not inflated** for these models.

---

### Ideal Mask (`la_roi_*` and `la_sub_*` metrics)
**`lung_arteries` only** — the pulmonary trunk, main right and left pulmonary arteries, lobar arteries, and segmental arteries. This is the direct anatomical site of PE.

**Status: Implemented.** Activate with `--la_eval` (reuses `--pe_roi_data_root`).

**Residual limitation:** TS misses subsegmental arteries (Dice ~0.94–0.97 on main/lobar, near-zero on subsegmental). The `la_*` metrics therefore still underrepresent the subsegmental regime. The global `sub_mae` remains the only metric capturing subsegmental performance without this blind spot.

---

## Tier 1 — Primary Endpoint: Subtraction Quality (Fairest Comparison)

### 1a — Global subtraction (no mask dependency)

These metrics directly answer the research question without depending on any imperfect vessel mask. They compare the synthetic subtraction image against the ground truth subtraction over the full volume, capturing all iodine-enhanced regions including subsegmental vessels that TS cannot segment.

**No inflation caveat for GAN baselines:** Unlike correction networks, GANs predict every voxel from scratch. The background is not trivially copied from the source, so `sub_psnr` and `sub_ssim` are directly comparable across architectures when evaluating GAN vs GAN.

**Flag:** `--subtraction_eval`

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `sub_ssim` | SSIM of subtraction image vs GT subtraction | ↑ | `--subtraction_eval` | No inflation for GAN baselines |
| `sub_mae` | MAE of subtraction image vs GT subtraction | ↓ | `--subtraction_eval` | Most honest for cross-architecture comparison |
| `sub_psnr` | PSNR of subtraction image vs GT subtraction | ↑ | `--subtraction_eval` | Not inflated for GANs |
| `sub_rmse` | RMSE of subtraction image vs GT subtraction | ↓ | `--subtraction_eval` | Redundant with sub_mae; log-scale sensitivity to outliers |

---

### 1b — Lung-arteries-only subtraction (`la_*`, ideal PE mask)

These metrics restrict evaluation to the TotalSegmentator `lung_arteries` label — the direct anatomical site of PE. They provide the most clinically specific single-structure comparison, and are now fully parity with the main repo.

**Flag:** `--la_eval` (requires `--pe_roi_data_root`)

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `la_sub_mae_hu` | MAE on subtraction within lung-arteries mask (HU scale) | ↓ | `--la_eval` | **Primary: most honest single metric for cross-architecture comparison within the PA** |
| `la_sub_ssim` | SSIM on subtraction within lung-arteries mask | ↑ | `--la_eval` | Per-slice SSIM averaged over slices with ≥10 mask voxels |
| `la_sub_psnr` | PSNR on subtraction within lung-arteries mask | ↑ | `--la_eval` | No background-copy inflation |
| `la_sub_mae_norm` | MAE on subtraction within lung-arteries mask (normalised) | ↓ | `--la_eval` | Normalised version of `la_sub_mae_hu` |
| `la_roi_cnr_pred` | CNR of predicted subtraction within lung-arteries mask | ↑ (target ≈ `la_roi_cnr_gt`) | `--la_eval` | Most clinically resonant: diagnostic signal in the exact anatomy where PE lodges |
| `la_roi_cnr_gt` | CNR of GT subtraction within lung-arteries mask | Reference | `--la_eval` | Use as reference for `la_roi_cnr_pred` |
| `la_roi_mae_hu` | MAE within lung-arteries mask (HU scale) | ↓ | `--la_eval` | VNC-quality metric restricted to PA |
| `la_roi_psnr` | PSNR within lung-arteries mask | ↑ | `--la_eval` | |
| `la_roi_ssim` | SSIM within lung-arteries mask | ↑ | `--la_eval` | |
| `la_roi_nmi` | NMI within lung-arteries mask | ↑ | `--la_eval` | |
| `la_roi_ncc` | NCC within lung-arteries mask | ↑ | `--la_eval` | |
| `la_roi_rmse` | RMSE within lung-arteries mask | ↓ | `--la_eval` | |
| `la_roi_dice` | Dice on lung-arteries mask | ↑ | `--la_eval` | Binary overlap at 50 HU threshold |
| `la_roi_mae` | MAE within lung-arteries mask (normalised) | ↓ | `--la_eval` | Normalised version of `la_roi_mae_hu` |

---

## Tier 2 — Secondary: Vessel-Region Subtraction Quality (Mask-Restricted)

These metrics restrict subtraction quality evaluation to the TS vessel mask. For GAN baselines, the "fair sub" concept (eliminating background-copy inflation) is less critical since GANs don't copy the background — but these metrics still provide a clinically relevant vessel-restricted comparison and enable direct cross-architecture comparison with correction networks.

**Flag:** `--pe_roi_eval` (requires `--pe_roi_data_root`)

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `fair_sub_mae_hu` | MAE on subtraction within TS mask (HU scale) | ↓ | `--pe_roi_eval` | Most honest single metric for correction-network cross-comparison |
| `fair_sub_ssim` | SSIM on subtraction within TS mask | ↑ | `--pe_roi_eval` | Structural quality of the subtraction inside vessels |
| `fair_sub_psnr` | PSNR on subtraction within TS mask | ↑ | `--pe_roi_eval` | |
| `fair_sub_mae_norm` | MAE on subtraction within TS mask (normalised) | ↓ | `--pe_roi_eval` | Normalised version of `fair_sub_mae_hu` |
| `pe_roi_cnr_pred` | CNR of predicted subtraction within TS mask | ↑ (target ≈ `pe_roi_cnr_gt`) | `--pe_roi_eval` | Clinical utility in broader vessel mask (incl. aorta, SVC, veins) |
| `pe_roi_cnr_gt` | CNR of GT subtraction within TS mask | Reference | `--pe_roi_eval` | Reference for cnr_pred |
| `pe_roi_mae_hu` | MAE within TS mask (HU scale) | ↓ | `--pe_roi_eval` | |
| `pe_roi_psnr` | PSNR within TS mask | ↑ | `--pe_roi_eval` | |
| `pe_roi_ssim` | SSIM within TS mask | ↑ | `--pe_roi_eval` | |
| `pe_roi_nmi` | NMI within TS mask | ↑ | `--pe_roi_eval` | |
| `pe_roi_ncc` | NCC within TS mask | ↑ | `--pe_roi_eval` | |
| `pe_roi_rmse` | RMSE within TS mask | ↓ | `--pe_roi_eval` | |
| `pe_roi_mae` | MAE within TS mask (normalised) | ↓ | `--pe_roi_eval` | |
| `pe_roi_dice` | Dice on TS vessel mask | ↑ | `--pe_roi_eval` | |

---

## Tier 3 — Secondary: Vessel-Region Non-Contrast CT Quality (Threshold Mask)

These metrics measure reconstruction quality within the threshold-based vessel mask. Useful for literature comparison and within-architecture analysis.

**Flag:** `--roi_eval`

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `roi_cnr_pred` | CNR of predicted subtraction within threshold mask | ↑ | `--roi_eval` | Useful when `pe_roi_eval` masks are unavailable |
| `roi_cnr_gt` | CNR of GT subtraction within threshold mask | Reference | `--roi_eval` | |
| `roi_mae_hu` | MAE within threshold mask (HU scale) | ↓ | `--roi_eval` | |
| `roi_psnr` | PSNR within threshold mask | ↑ | `--roi_eval` | |
| `roi_ssim` | SSIM within threshold mask | ↑ | `--roi_eval` | |
| `roi_nmi` | NMI within threshold mask | ↑ | `--roi_eval` | |
| `roi_ncc` | NCC within threshold mask | ↑ | `--roi_eval` | |
| `roi_rmse` | RMSE within threshold mask | ↓ | `--roi_eval` | |
| `roi_mae` | MAE within threshold mask (normalised) | ↓ | `--roi_eval` | |
| `roi_dice` | Dice on threshold vessel mask | ↑ | `--roi_eval` | |

---

## Tier 4 — Supporting: Global Non-Contrast CT Quality

Standard metrics required for literature comparison. Always computed (no flag required).

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `ssim` | 3D SSIM (Gaussian kernel) | ↑ | always | |
| `ms_ssim` | Multi-Scale SSIM (5-scale pyramid) | ↑ | always | More robust to registration misalignment than single-scale SSIM |
| `psnr` | Peak Signal-to-Noise Ratio | ↑ | always | |
| `mae` | Mean Absolute Error (normalised) | ↓ | always | |
| `rmse` | Root Mean Squared Error | ↓ | always | |
| `mse` | Mean Squared Error | ↓ | always | |
| `nmi` | Normalised Mutual Information | ↑ | always | |
| `ncc` | Normalised Cross-Correlation | ↑ | always | |

**Not available in this repo (available in main repo's `evaluate.py` / `evaluate_pixel.py`):**

| Metric | Reason absent |
|---|---|
| `ssim_3d_np` | NumPy uniform-filter SSIM variant — not implemented in `test.py` |

---

## Tier 5 — Distributional: Texture Realism (Dataset-Level)

Computed once over the full test set. Reveal systematic texture or style failures invisible to per-patient pixel metrics.

| Metric | Full Name | Direction | Flag | Notes |
|---|---|---|---|---|
| `frd` | Fréchet Radiomic Distance | ↓ | `--frd_eval` | **Preferred over FID for medical imaging.** Uses 22 GLCM texture features via PyRadiomics. |
| `fid_2d` | Fréchet Inception Distance (2D) | ↓ | `--fid_eval` | InceptionV3 features on centre axial slice. Less appropriate than FRD for CT but universally reported. |

---

## Tier 6 — Alignment / Tier 7 — Mechanistic Diagnostics

**Not available in this repo.** The main repo's `evaluate.py` (`--alignment_analysis`) and `evaluate_correction_3d.py` (`--correction_diagnostics`) implement these. They are specific to the diffusion model and correction network architectures.

| Metric group | Reason absent |
|---|---|
| `nmi_fixed_gen`, `ncc_fixed_gen`, `ssim_fixed_gen` | `--alignment_analysis` not implemented in `test.py` |
| `mask_coverage`, `background_leakage`, `cnr_preservation`, `vessel_ssim`, `enhancement_sparsity` | Correction-network diagnostics; not applicable to GANs |

---

## Summary Table: What to Report in a Paper

| Priority | Metric(s) | Flag | Notes |
|---|---|---|---|
| Primary (global) | `sub_ssim`, `sub_mae` | `--subtraction_eval` | Fair subtraction quality across all vessels including subsegmental |
| Primary (PA-specific) | `la_sub_mae_hu`, `la_sub_ssim` | `--la_eval` | Subtraction quality restricted to pulmonary arteries; no background inflation |
| Primary (PA-specific) | `la_roi_cnr_pred` (vs `la_roi_cnr_gt`) | `--la_eval` | Diagnostic signal in the exact anatomy where PE lodges |
| Secondary | `fair_sub_mae_hu`, `fair_sub_ssim` | `--pe_roi_eval` | Vessel-restricted comparison; directly comparable to correction networks |
| Secondary | `pe_roi_cnr_pred` (vs `pe_roi_cnr_gt`) | `--pe_roi_eval` | Clinical utility in broader vessel mask (incl. aorta, SVC, veins) |
| Secondary | `ssim`, `ms_ssim`, `psnr` | always | Literature comparison (VNC papers) |
| Secondary | `roi_mae_hu`, `pe_roi_mae_hu` | `--roi_eval` / `--pe_roi_eval` | Vessel-region reconstruction; cite TS blind spot caveat |
| Distributional | `frd`, `fid_2d` | `--frd_eval` / `--fid_eval` | Texture realism; explain FRD > FID preference for CT |

**Remaining gap vs main repo (single metric):**

| Missing metric | Available in | Impact |
|---|---|---|
| `ssim_3d_np` | `evaluate.py`, `evaluate_pixel.py` | NumPy SSIM variant for cross-validation; low priority |

---

## Example Commands

```bash
# Full evaluation with all available metric groups
python test.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_cyclegan_baseline --model cycle_gan \
    --input_nc 1 --output_nc 1 --epoch best --eval \
    --subtraction_eval \
    --pe_roi_eval --pe_roi_data_root ../data/TS_Vessel_Masks \
    --la_eval \
    --roi_eval \
    --frd_eval --frd_num_slices 5 \
    --fid_eval

# Pix2Pix — same flags, different model
python test.py \
    --dataroot ../data/Coltea_Processed_Nifti_Registered \
    --name coltea_pix2pix_baseline --model pix2pix \
    --input_nc 1 --output_nc 1 --epoch best --eval \
    --subtraction_eval \
    --pe_roi_eval --pe_roi_data_root ../data/TS_Vessel_Masks \
    --la_eval \
    --frd_eval
```
