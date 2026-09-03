"""
Creates side-by-side comparison strips (NoisyLR | GT | Model Output) for the
N images where the model makes the biggest visual difference (highest PSNR gain
over the raw bicubic baseline).
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

HERE = os.path.dirname(os.path.abspath(__file__))

NOISY_DIR  = os.path.join(HERE, "data", "NoisyLR")
GT_DIR     = os.path.join(HERE, "data", "GT")
OUTPUT_DIR = os.path.join(HERE, "data", "output")
SAVE_DIR   = os.path.join(HERE, "comparisons")
TOP_N      = 5   # number of examples to show

os.makedirs(SAVE_DIR, exist_ok=True)

files = sorted(f for f in os.listdir(NOISY_DIR) if f.endswith(".npy")
               and os.path.exists(os.path.join(GT_DIR, f))
               and os.path.exists(os.path.join(OUTPUT_DIR, f)))

print(f"Scoring {len(files)} images for PSNR gain...", flush=True)

def load_npy(path):
    a = np.load(path).astype(np.float32)
    if a.ndim == 3 and a.shape[0] == 1: a = a[0]
    if a.ndim == 3 and a.shape[2] == 1: a = a[..., 0]
    return np.clip(a, 0.0, 1.0)

def bicubic_up(arr_hw):
    """Upsample H x W array 2x via bicubic."""
    t = torch.from_numpy(arr_hw).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, scale_factor=2, mode="bicubic", align_corners=False)
    return np.clip(t.squeeze().numpy(), 0.0, 1.0)

scores = []
for fname in files:
    noisy = load_npy(os.path.join(NOISY_DIR,  fname))
    gt    = load_npy(os.path.join(GT_DIR,     fname))
    pred  = load_npy(os.path.join(OUTPUT_DIR, fname))

    # align gt to pred shape if needed
    if gt.shape != pred.shape:
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        gt   = np.clip(F.interpolate(gt_t, size=pred.shape, mode="bicubic",
                                     align_corners=False).squeeze().numpy(), 0, 1)

    bicubic = bicubic_up(noisy)
    psnr_bicubic = ski_psnr(gt, bicubic, data_range=1.0)
    psnr_model   = ski_psnr(gt, pred,    data_range=1.0)
    gain = psnr_model - psnr_bicubic
    scores.append((gain, psnr_bicubic, psnr_model, fname))

scores.sort(reverse=True)   # highest gain first
top = scores[:TOP_N]

print(f"Top {TOP_N} by PSNR gain (model vs bicubic baseline):")
for g, pb, pm, f in top:
    print(f"  {f}  bicubic={pb:.2f}dB  model={pm:.2f}dB  gain=+{g:.2f}dB")

# ── build comparison images ──────────────────────────────────────────────────
PAD   = 8    # pixels between panels
LABEL_H = 28  # height of text label strip
BG    = 30   # dark background colour

def arr_to_uint8(a):
    return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)

def make_label(text, w, h=LABEL_H, bg=BG, fg=220):
    img = Image.new("L", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    # use default font — works on any system
    draw.text((4, 4), text, fill=fg)
    return np.array(img)

saved_paths = []
for rank, (gain, psnr_bic, psnr_mod, fname) in enumerate(top, 1):
    noisy = load_npy(os.path.join(NOISY_DIR,  fname))
    gt    = load_npy(os.path.join(GT_DIR,     fname))
    pred  = load_npy(os.path.join(OUTPUT_DIR, fname))

    bicubic = bicubic_up(noisy)

    if gt.shape != pred.shape:
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        gt   = np.clip(F.interpolate(gt_t, size=pred.shape, mode="bicubic",
                                     align_corners=False).squeeze().numpy(), 0, 1)

    H, W = pred.shape   # 256 x 256

    panels = [
        ("NoisyLR (input)", bicubic),   # show bicubic-upsampled for fair size comparison
        ("Ground Truth",    gt),
        ("Model Output",    pred),
    ]

    strip_w = W * 3 + PAD * 4
    strip_h = LABEL_H + H + PAD * 2

    canvas = np.full((strip_h, strip_w), BG, dtype=np.uint8)

    x = PAD
    for label, arr in panels:
        lbl = make_label(label, W)
        canvas[PAD : PAD + LABEL_H, x : x + W] = lbl
        canvas[PAD + LABEL_H : PAD + LABEL_H + H, x : x + W] = arr_to_uint8(arr)
        x += W + PAD

    # title bar at top
    img = Image.fromarray(canvas, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    title = f"#{rank}  {fname}  |  Bicubic {psnr_bic:.2f}dB  ->  Model {psnr_mod:.2f}dB  (gain +{gain:.2f}dB)"
    # draw a dark title bar
    draw.rectangle([0, 0, strip_w, LABEL_H - 2], fill=(20, 20, 20))
    draw.text((6, 5), title, fill=(200, 220, 255))

    out_path = os.path.join(SAVE_DIR, f"comparison_{rank:02d}_{os.path.splitext(fname)[0]}.png")
    img.save(out_path)
    saved_paths.append(out_path)
    print(f"  Saved: {out_path}")

# ── also make one combined grid of all examples ───────────────────────────────
strips = [np.array(Image.open(p).convert("RGB")) for p in saved_paths]
grid   = np.concatenate(strips, axis=0)
grid_path = os.path.join(SAVE_DIR, "top5_comparison_grid.png")
Image.fromarray(grid.astype(np.uint8)).save(grid_path)
print(f"\nCombined grid saved: {grid_path}")
