"""Plain white-background visual comparison grid: Bicubic | Adaptive Wiener
(best classical) | Shipped model | Noise-map variant (experimental) | GT,
for a handful of examples chosen for visually-evident differences between
methods. No branded colors/background -- plain white figure, matplotlib
defaults.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESTORATION = os.path.join(ROOT, "restoration")
sys.path.insert(0, RESTORATION)

from evaluate import load_model  # noqa: E402
from losses import estimated_noise_variance  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(os.path.join(RESTORATION, "data_manifest.json")))
gt_dir, nlr_dir = manifest["gt_dir"], manifest["noisylr_dir"]
val_files = manifest["val_files"]


def load_pair(fname):
    gt = np.load(os.path.join(gt_dir, fname)).astype(np.float32)
    nlr = np.load(os.path.join(nlr_dir, fname)).astype(np.float32)
    bicubic = resize(nlr, gt.shape, order=3, mode="edge", anti_aliasing=False).astype(np.float32)
    return gt, nlr, bicubic


def adaptive_wiener(bicubic, win=5):
    """Local-statistics (Lee-type) adaptive Wiener filter using our own
    measured noise-variance-vs-intensity curve as the noise-power term."""
    local_mean = uniform_filter(bicubic, size=win)
    local_sqmean = uniform_filter(bicubic ** 2, size=win)
    local_var = np.clip(local_sqmean - local_mean ** 2, 0, None)
    noise_var = estimated_noise_variance(torch.from_numpy(local_mean)).numpy()
    gain = np.clip(local_var - noise_var, 0, None) / np.clip(local_var, 1e-6, None)
    out = local_mean + gain * (bicubic - local_mean)
    return np.clip(out, 0.0, 1.0)


def run_model(model, bicubic, use_noise_map=False):
    with torch.no_grad():
        x = torch.from_numpy(bicubic).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
    return out.float().cpu().numpy()[0, 0]


print("Loading models...")
shipped_model, _ = load_model("nafnet_full", os.path.join(RESTORATION, "runs", "main_run_6h", "best.pth"), device)
noisemap_model, _ = load_model("nafnet_full", os.path.join(RESTORATION, "runs", "medtest_noisemap", "best.pth"),
                                device, use_noise_map=True)

# Pick examples spanning a range of bicubic difficulty for visually evident
# differences: scan a subset of val files, keep 3 with the lowest bicubic
# PSNR (hardest -- where method differences show up most) and 1 easy case.
print("Scanning val files for bicubic difficulty...")
scored = []
for fname in val_files[:200]:
    gt, nlr, bicubic = load_pair(fname)
    p = peak_signal_noise_ratio(gt, bicubic, data_range=1.0)
    scored.append((fname, p))
scored.sort(key=lambda t: t[1])

chosen = [scored[5][0], scored[25][0], scored[60][0], scored[-15][0]]
print("Chosen examples:", chosen)

methods = ["Bicubic\n(baseline)", "Adaptive Wiener\n(best classical)", "Shipped Model", "Noise-map variant\n(experimental)", "Ground Truth"]
n_rows = len(chosen)
n_cols = len(methods)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.4 * n_rows), facecolor="white")

for r, fname in enumerate(chosen):
    gt, nlr, bicubic = load_pair(fname)
    wiener_out = adaptive_wiener(bicubic)
    shipped_out = run_model(shipped_model, bicubic)
    noisemap_out = run_model(noisemap_model, bicubic)

    imgs = [bicubic, wiener_out, shipped_out, noisemap_out, gt]
    for c, (img, name) in enumerate(zip(imgs, methods)):
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if name != "Ground Truth":
            p = peak_signal_noise_ratio(gt, np.clip(img, 0, 1), data_range=1.0)
            s = structural_similarity(gt, np.clip(img, 0, 1), data_range=1.0)
            ax.set_title(f"{name}\nPSNR {p:.1f}  SSIM {s:.3f}", fontsize=9.5, color="black")
        else:
            ax.set_title(name, fontsize=9.5, color="black")
        if c == 0:
            ax.set_ylabel(fname.replace(".npy", ""), fontsize=8, color="dimgray")

fig.suptitle("Visual Comparison: Classical Baseline vs. Shipped Model vs. Experimental Variant",
             fontsize=13, color="black")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_dir = os.path.join(HERE, "08_visual_comparisons")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "full_pipeline_visual_comparison.png")
fig.savefig(out_path, dpi=150, facecolor="white")
plt.close(fig)
print("wrote", out_path)
