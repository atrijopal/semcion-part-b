# KLA AI Hackathon — Image Restoration Submission

Joint denoising + 2× super-resolution for grayscale SEM (Scanning Electron
Microscopy) inspection imagery. This folder contains all four required
submission components (per the problem statement, Slide 17) plus a
`benchmarks/` folder documenting how the model was designed, tested, and
evaluated.

## Contents

```
SUBMISSION/
├── README.md             this file
├── evaluate.py           required component 1 -- evaluation script
├── model.py              model architecture (imported by evaluate.py)
├── weights/
│   └── model_best.pth    trained checkpoint (29.07M params)
├── training/             required component 2 -- training script
│   └── ...               (see training/README.md)
├── outputs/               required component 3 -- denoised test outputs
│   └── *.npy              400 files, model output on the provided test set
├── requirements.txt      required component 4 -- environment specification
└── benchmarks/            data analysis, architecture selection,
                            and evaluation methodology
```

## 1. Evaluation script

```
python evaluate.py <input_dir> <output_dir>
```

Standalone, non-notebook script. No manual edits required — model
architecture and trained weights are located via a path relative to the
script itself, so it runs correctly regardless of working directory.
Validated end-to-end from an independent working directory prior to
submission.

Reads every `.npy` file in `<input_dir>` (single-channel float32, any
resolution — the model performs 2× super-resolution, so a 128×128 input
produces a 256×256 output) and writes a same-named `.npy` restored image
to `<output_dir>`, clipped to `[0, 1]`.

Measured end-to-end runtime on the full 400-image test set (RTX 4050
Laptop GPU): **9.3s total** (model initialization 2.2s; inference + I/O
7.1s, ≈17.8 ms/image).

## 2. Training script

See `training/README.md`. Fully self-contained — no external repo clone or
download needed; the NAFNet building blocks and the NAFNet-SIDD pretrained
checkpoint are bundled directly.

## 3. Denoised test outputs

`outputs/` — this submission's `evaluate.py`, run against the provided
`Test_NoisyLR` set (400 images, no ground truth), output as-is.

## 4. Environment specification

`requirements.txt` — full `pip freeze` from the environment used to train
and evaluate this submission (Python 3.14.4, PyTorch 2.13.0+cu130).

## Model summary

- **Architecture**: NAFNet-full (Chen et al., "Simple Baselines for Image
  Restoration"), 29.07M parameters, additive-residual joint denoise + 2×
  super-resolution, bias-free convolutions, initialized from the official
  NAFNet-SIDD pretrained checkpoint (channel-adapted for grayscale input).
  Selected via controlled, equal-budget evaluation against a lightweight
  transformer alternative — see `benchmarks/architecture_bakeoff.json`.
- **Loss function**: Charbonnier + SSIM + an edge/Laplacian term + LPIPS
  (warmup schedule). Loss weights were set via controlled comparison rather
  than literature defaults — see `benchmarks/BENCHMARKS_REPORT.md`.
- **Training data**: the provided training set, with 30 images excluded
  after being confirmed to have corrupted ground truth (sensor noise, no
  real structure) via a validated detection method — see `benchmarks/eda/`.
- **Final validation metrics** (477-image held-out split):
  **PSNR 23.53 dB, SSIM 0.622, LPIPS 0.143**, 18.2 ms/image inference.
  Full results and methodology in `benchmarks/BENCHMARKS_REPORT.md`.
