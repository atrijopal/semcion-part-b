# Training script

Reproduces the training of the submitted model (`../weights/model_best.pth`).
Self-contained — no external repo clone needed; the NAFNet building blocks
(`model_nafnet.py`) and the NAFNet-SIDD pretrained checkpoint
(`pretrained/NAFNet-SIDD-width32.pth`) are bundled directly.

## Setup

1. Install dependencies: `pip install -r ../requirements.txt` (or at minimum
   `torch`, `torchvision`, `numpy`, `opencv-python`, `scikit-image`, `lpips`).
2. Place the official training dataset at
   `training/data/semicon_train_data/semicon_train_data/{GT,NoisyLR}`
   (matching the structure it's distributed in), or edit `gt_dir` /
   `noisylr_dir` in `data_manifest.json` to point elsewhere.

`data_manifest.json` is already the resolved train/val split (4278 train /
477 val, stratified by unsupervised texture clustering, with 30 confirmed-
corrupted GT images excluded — see `../benchmarks/eda/` for how this was
derived). `data_manifest.py` is included for reference/transparency on that
derivation only — it depends on this project's original working directory
and is **not** meant to be re-run as part of reproducing training.

## Run

```
python train.py --arch nafnet_full --run_name my_run --budget_minutes 360 \
    --lam_ssim 0.15 --lam_edge 0.08 --lam_lpips 0.08
```

This is the configuration used for the submitted checkpoint. Architecture
and hyperparameter selection methodology is documented in
`../benchmarks/BENCHMARKS_REPORT.md`.

Key flags:
- `--budget_minutes`: wall-clock training budget. Progressive patch size
  (64px for the first 60%, 128px for the last 40%) and a cosine LR decay
  are both tied to elapsed-time-vs-budget, not epoch count.
- `--resume <checkpoint path>`: restores model/optimizer/scaler state and
  elapsed-time-so-far, so an interrupted run continues rather than
  restarting. `runs/<run_name>/periodic.pth` is checkpointed every 5 minutes
  by default (`--ckpt_every_sec`); `runs/<run_name>/best.pth` tracks the
  highest-val-PSNR checkpoint seen at any point.
- `--no_pretrained`: train from random init instead of the NAFNet-SIDD
  checkpoint. Pretrained init is used for the submitted model; this flag is
  provided for ablation.

Live progress: `runs/<run_name>/status.json` is rewritten every epoch
(state/epoch/elapsed/budget/loss/val metrics/last checkpoint path) — safe
to poll from another process. `runs/<run_name>/history.json` has the full
per-epoch training curve.

## Evaluate a trained checkpoint

```
python evaluate.py --checkpoint runs/my_run/best.pth --arch nafnet_full
```

Computes overall PSNR/SSIM/LPIPS and measured (GPU-warmed-up) inference time
on the manifest's held-out validation split. This is the internal
metrics/validation script (used throughout development against a
ground-truth-bearing validation split) — distinct from `../run.py`,
the submission's required inference-only script, which has no metrics
computation since the actual competition test set has no ground truth.

## Files

- `train.py` — the training loop (config-driven, wall-clock budget,
  checkpointing, resume, live status).
- `evaluate.py` — validation metrics script (PSNR/SSIM/LPIPS/latency),
  requires a manifest with ground truth.
- `model_nafnet.py` — NAFNet-full architecture (same as `../model.py`).
- `model_swinir_lite.py` — the alternative architecture evaluated during
  model selection (`--arch swinir_lite`), included for reproducibility of
  the comparison in `../benchmarks/BENCHMARKS_REPORT.md`.
- `dataset.py` — data loading, real+synthetic augmentation mixing,
  progressive patch cropping.
- `degradation_sim.py` — physics-informed synthetic degradation generator
  (Gamma-distributed speckle + Gaussian noise + randomized downsample
  kernel, shuffled order) used to augment training beyond the real pairs.
- `losses.py` — Charbonnier + SSIM (computed in fp32 regardless of mixed-
  precision context, required for gradient stability) + edge/Laplacian +
  LPIPS-with-warmup, each independently toggleable.
- `pretrained_loader.py` — loads and channel-adapts the bundled NAFNet-SIDD
  checkpoint for 1-channel (grayscale) input/output.
- `data_manifest.json` / `data_manifest.py` — the train/val split (see above).
- `corrupted_gt_exclusion.json` — the 30 excluded files + the detection
  method (max lag-1 spatial autocorrelation over 32×32 blocks — a
  genuinely corrupted GT patch has no block anywhere with real structure).
