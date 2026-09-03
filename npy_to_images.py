"""
Convert all .npy files in a directory to PNG images.

Usage:
    python npy_to_images.py                         # converts outputs/ → images/
    python npy_to_images.py <input_dir>             # custom npy dir → images/
    python npy_to_images.py <input_dir> <out_dir>   # custom both
"""
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ── args ─────────────────────────────────────────────────────────────────────
npy_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data/output")
img_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "output_images")

os.makedirs(img_dir, exist_ok=True)

files = sorted(f for f in os.listdir(npy_dir) if f.endswith(".npy"))
print(f"Found {len(files)} .npy files in '{npy_dir}'")
print(f"Saving PNG images to  '{img_dir}'\n")

for i, fname in enumerate(files):
    arr = np.load(os.path.join(npy_dir, fname)).astype(np.float32)

    # handle (H,W,1) or (1,H,W) shapes defensively
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    # clip to [0,1] and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    img_arr = (arr * 255).round().astype(np.uint8)

    img = Image.fromarray(img_arr, mode="L")   # "L" = grayscale

    out_name = os.path.splitext(fname)[0] + ".png"
    img.save(os.path.join(img_dir, out_name))

    if (i + 1) % 50 == 0 or (i + 1) == len(files):
        print(f"  [{i+1:>4}/{len(files)}]  {fname}  ->  {out_name}  ({img_arr.shape[1]}x{img_arr.shape[0]}px)")

print(f"\nDone. {len(files)} images saved to '{img_dir}'")
