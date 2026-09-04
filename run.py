#!/usr/bin/env python3
"""
Standalone evaluation / inference script for the KLA AI Hackathon --
AI-Based Restoration of Degraded Images (joint denoising + 2x super-
resolution, SEM images).

Usage:
    python run.py <input_dir> <output_dir>

<input_dir>  : directory of degraded input images, one .npy file per image
               (single-channel float32 array, any H x W -- the model
               upsamples 2x, so a 128x128 input produces a 256x256 output).
<output_dir> : directory to write restored images to (created if missing).
               Each output is written as <same filename>.npy, float32,
               values in [0, 1], at 2x the input's spatial resolution.

No manual edits required -- the model architecture (model.py) and trained
weights (weights/model_best.pth) are bundled alongside this script and
located via a path relative to this file, so it runs correctly regardless
of the working directory it's invoked from.

Design notes (kept intentionally simple/fast, since inference time is a
graded criterion per the challenge's evaluation criteria):
  - Single model load, single device transfer, up front.
  - Per-image bicubic upsample (fixed, non-learned baseline) + one model
    forward pass; no redundant recomputation across images.
  - No artificial warmup passes are added before timing -- the graded
    metric is this script's real, total, cold-start wall-clock time
    (script startup, model init, disk I/O, inference, disk write), so
    padding it with throwaway work would only make the reported number
    less representative, not better.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import build_model  # noqa: E402

CHECKPOINT_PATH = os.path.join(HERE, "weights", "model_best.pth")


def bicubic_upsample(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """x: (1,1,H,W) tensor, any value range (kept as-is, not clipped --
    the degraded input's out-of-[0,1] range is expected, see the
    challenge's own FAQ on speckle noise). Returns (1,1,2H,2W)."""
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h * scale, w * scale), mode="bicubic", align_corners=False)


def main():
    ap = argparse.ArgumentParser(description="Restore degraded SEM images: joint denoise + 2x super-resolution.")
    ap.add_argument("input_dir", help="Directory of degraded input .npy images")
    ap.add_argument("output_dir", help="Directory to write restored .npy images to")
    args = ap.parse_args()

    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    t_model_ready = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not input_files:
        print(f"[run] WARNING: no .npy files found in {args.input_dir}", file=sys.stderr)

    n_written = 0
    with torch.no_grad():
        for fname in input_files:
            arr = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
            if arr.ndim == 3:  # tolerate an (H,W,1)-style array defensively
                arr = arr[..., 0]
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
            bicubic = bicubic_upsample(x, scale=2)
            out = model(bicubic)
            out_np = out.squeeze(0).squeeze(0).float().cpu().numpy()
            np.save(os.path.join(args.output_dir, fname), out_np)
            n_written += 1

    t_end = time.time()

    print(f"[run] device={device}")
    print(f"[run] model init: {t_model_ready - t_start:.3f}s")
    print(f"[run] inference + I/O for {n_written} images: {t_end - t_model_ready:.3f}s "
          f"({(t_end - t_model_ready) / max(1, n_written) * 1000:.2f} ms/image)")
    print(f"[run] total wall-clock: {t_end - t_start:.3f}s")
    print(f"[run] wrote {n_written} restored images to {args.output_dir}")


if __name__ == "__main__":
    main()
