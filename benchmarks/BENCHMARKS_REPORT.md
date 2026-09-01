# Benchmarks & Design Report

## 1. Data Analysis

Exploratory analysis of the full training set (4785 GT/NoisyLR pairs) is
in `eda/eda_report.md`, with supporting figures in `eda/figures/`. Findings
that informed model design:

| Finding | Design implication |
|---|---|
| NoisyLR intensity exceeds [0,1], content-dependently (99.1% of images) | Input is not clipped before the network; only the final output is clamped to [0,1] |
| Noise variance scales with local intensity (signal-dependent) | Motivates the Charbonnier + SSIM loss combination over plain MSE |
| Laplacian variance (sharpness) of degraded inputs is ~0.35x ground truth, despite *higher* raw gradient energy (noise-inflated) | Added an explicit edge/Laplacian loss term |
| 30 images have corrupted ground truth (sensor noise, no real structure) — detected via spatial-autocorrelation analysis, confirmed visually | Excluded from training (`corrupted_gt_exclusion.json`) |
| Bicubic-only baseline: 20.3 dB PSNR / 0.510 SSIM | Reference floor for all reported gains |

## 2. Architecture Selection

Two candidates were evaluated under identical conditions (same data split,
same loss configuration, equal training budget):

| Architecture | Params | PSNR | SSIM | LPIPS | Inference (eager) | Inference (compiled) |
|---|---|---|---|---|---|---|
| **NAFNet (selected)** | 29.07M | **23.28 dB** | **0.611** | **0.222** | **16.3 ms** | **2.7 ms** |
| SwinIR-lightweight | 1.03M | 21.87 dB | 0.573 | 0.328 | 61.0 ms | 38.8 ms |

NAFNet was selected: it leads on every measured metric, including
inference latency despite its larger parameter count. Full methodology and
per-metric results in `architecture_bakeoff.json`.

A dual-residual fusion variant (multiplicative + additive branches,
targeting speckle noise explicitly) was evaluated in prior related work and
did not pass its pre-registered validation criteria; a single additive-
residual design was used instead.

## 3. Loss Configuration

Final loss: `Charbonnier + 0.15 × SSIM + 0.08 × Edge(Laplacian) + 0.08 × LPIPS (warmup)`.

The SSIM weight was set via controlled comparison rather than adopted from
a generic literature default:

| SSIM weight | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 0.84 | 23.38 dB | 0.589 | 0.265 |
| **0.15 (selected)** | 23.43 dB | 0.587 | **0.189** |

SSIM was statistically equivalent between the two configurations; the lower
weight improved LPIPS by ~29% and was selected for the final training run.

## 4. Training Configuration

NAFNet, pretrained (NAFNet-SIDD) initialization, on the deduplicated,
cluster-stratified training split (4278 train / 477 validation). Progressive
patch scheduling (64px → 128px), cosine learning-rate decay over the
training budget, checkpointed every 5 minutes with full resume support.

Training ran for 600 epochs (≈313 minutes) before being concluded, following
a sustained validation-metric plateau confirmed across multiple independent
checks.

## 5. Final Model Performance

Evaluated on the full 477-image held-out validation split
(`final_model_metrics.json`):

| Metric | Value |
|---|---|
| PSNR | 23.53 dB |
| SSIM | 0.622 |
| LPIPS | 0.143 |
| Inference latency | 18.2 ms/image |

`torch.compile` was evaluated for the submitted inference script and not
adopted: on the actual test-set size (400 images), its fixed compilation
overhead exceeds the runtime saved, producing a net increase in total
wall-clock time (23.7s compiled vs. 7.5s eager for the full test set).

## 6. Restoration Quality by Content Type

Visual inspection across the full validation quality range
(`quality_examples/`, 30 images spanning worst to best) shows restoration
quality is strongly content-dependent:

- **Large-scale structural content** (particles, blobs, cavities): restored
  close to ground truth (32–34 dB range).
- **Fine, high-frequency mesh/fibrous textures**: consistently under-
  restored (17–19 dB range) — the model smooths detail that ground truth
  retains.

This pattern is consistent with information loss inherent to the
degradation process (2x downsampling removes high-frequency content that
cannot be fully recovered by any single-pass regression model) rather than
an architecture-specific limitation.
