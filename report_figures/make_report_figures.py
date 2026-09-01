"""Generates the three new figures needed for FULL_TECHNICAL_REPORT.md that
don't already exist in eda/outputs/figures or dataset_comparison/figures:
1. Training curve (main_run_6h, full 600 epochs)
2. Old model vs new model, on old and new datasets (grouped bar chart)
3. All-experiments summary (every tested configuration, one chart)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# 1. Training curve
# ---------------------------------------------------------------------------
history = json.load(open(os.path.join(ROOT, "restoration", "runs", "main_run_6h", "history.json")))
val_entries = [e for e in history if e.get("val_psnr") is not None]
epochs = [e["epoch"] for e in val_entries]
psnr = [e["val_psnr"] for e in val_entries]
ssim = [e["val_ssim"] for e in val_entries]

fig, ax1 = plt.subplots(figsize=(11, 5.5))
ax2 = ax1.twinx()

l1, = ax1.plot(epochs, psnr, color="tab:blue", linewidth=1.8, label="Validation PSNR (dB)")
ax1.axhline(20.22, color="tab:blue", linestyle=":", linewidth=1.3, alpha=0.7)
ax1.text(epochs[-1] * 0.02, 20.5, "bicubic-only baseline (20.22 dB)", color="tab:blue", fontsize=9, alpha=0.85)

l2, = ax2.plot(epochs, ssim, color="tab:orange", linewidth=1.8, label="Validation SSIM")
ax2.axhline(0.503, color="tab:orange", linestyle=":", linewidth=1.3, alpha=0.7)
ax2.text(epochs[-1] * 0.02, 0.483, "bicubic-only baseline (0.503)", color="tab:orange", fontsize=9, alpha=0.85)

best_epoch, best_psnr = 583, 23.703
ax1.scatter([best_epoch], [best_psnr], color="black", zorder=5, s=45)
ax1.annotate(f"selected checkpoint\nepoch {best_epoch}, {best_psnr:.2f} dB",
             xy=(best_epoch, best_psnr), xytext=(best_epoch - 220, best_psnr - 1.9),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=1))

ax1.set_xlabel("Epoch")
ax1.set_ylabel("PSNR (dB)", color="tab:blue")
ax2.set_ylabel("SSIM", color="tab:orange")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax1.set_title("Production Training Run — Validation Metrics over 600 Epochs (~5.2h, RTX 4050)")
ax1.grid(alpha=0.25)
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "01_training_curve_full.png"), dpi=150)
plt.close(fig)
print("wrote 01_training_curve_full.png")

# ---------------------------------------------------------------------------
# 2. Old model vs new model, old dataset vs new dataset
# ---------------------------------------------------------------------------
groups = ["Old model /\nold dataset", "New model /\nold dataset", "Old model /\nnew dataset", "New model /\nnew dataset"]
psnr_vals = [27.772, 28.264, 23.061, 23.526]
ssim_vals = [0.742, 0.756, 0.568, 0.622]
lpips_vals = [None, 0.191, 0.446, 0.143]

x = np.arange(len(groups))
width = 0.35

fig, (axp, axs) = plt.subplots(1, 2, figsize=(13, 5.5))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
bars = axp.bar(x, psnr_vals, color=colors, width=0.55)
axp.set_ylabel("PSNR (dB)")
axp.set_title("PSNR — Old vs. New Model, Old vs. New Dataset")
axp.set_xticks(x)
axp.set_xticklabels(groups, fontsize=9)
axp.set_ylim(0, 32)
for b, v in zip(bars, psnr_vals):
    axp.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=9)
axp.grid(axis="y", alpha=0.25)

bars2 = axs.bar(x, ssim_vals, color=colors, width=0.55)
axs.set_ylabel("SSIM")
axs.set_title("SSIM — Old vs. New Model, Old vs. New Dataset")
axs.set_xticks(x)
axs.set_xticklabels(groups, fontsize=9)
axs.set_ylim(0, 0.9)
for b, v in zip(bars2, ssim_vals):
    axs.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
axs.grid(axis="y", alpha=0.25)

fig.suptitle("Note: LPIPS not recorded for old model on the old dataset (not measured in prior project); "
              "LPIPS on new dataset: old model 0.446, new model 0.143", fontsize=8.5, y=0.01)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(HERE, "02_old_vs_new_model.png"), dpi=150)
plt.close(fig)
print("wrote 02_old_vs_new_model.png")

# ---------------------------------------------------------------------------
# 3. All-experiments summary
# ---------------------------------------------------------------------------
configs = [
    ("Bicubic\n(baseline)", 20.22, 0.503, None, "tab:gray"),
    ("Best classical\n(Wiener)", 22.43, 0.545, None, "tab:purple"),
    ("Weighted loss\nv1", 23.26, 0.584, 0.194, "tab:orange"),
    ("Weighted loss\nv2", 23.12, 0.588, 0.184, "tab:orange"),
    ("Noise-map\ninput", 23.634, 0.6123, 0.171, "tab:orange"),
    ("SHIPPED\nMODEL", 23.643, 0.6253, 0.1362, "tab:green"),
]
labels = [c[0] for c in configs]
psnr_v = [c[1] for c in configs]
ssim_v = [c[2] for c in configs]
colors3 = [c[4] for c in configs]

fig, (axp, axs) = plt.subplots(1, 2, figsize=(15, 6.2))
x = np.arange(len(labels))

barsp = axp.bar(x, psnr_v, color=colors3, width=0.6)
axp.set_ylabel("PSNR (dB)")
axp.set_title("PSNR — All Tested Configurations\n(full 4,755-image set unless noted)")
axp.set_xticks(x)
axp.set_xticklabels(labels, fontsize=9.5)
axp.set_ylim(18, 25.5)
for b, v in zip(barsp, psnr_v):
    axp.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
axp.grid(axis="y", alpha=0.25)

barss = axs.bar(x, ssim_v, color=colors3, width=0.6)
axs.set_ylabel("SSIM")
axs.set_title("SSIM — All Tested Configurations")
axs.set_xticks(x)
axs.set_xticklabels(labels, fontsize=9.5)
axs.set_ylim(0.45, 0.68)
for b, v in zip(barss, ssim_v):
    axs.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
axs.grid(axis="y", alpha=0.25)

fig.suptitle("Every improvement attempt tested against the shipped model — none surpassed it", fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.text(0.5, 0.01,
          "Weighted loss v1/v2, Noise-map input: equal-budget short controlled tests (12–40 min), not full 5.2h runs — see report text.",
          ha="center", fontsize=8, style="italic")
fig.savefig(os.path.join(HERE, "03_all_experiments_summary.png"), dpi=150)
plt.close(fig)
print("wrote 03_all_experiments_summary.png")
