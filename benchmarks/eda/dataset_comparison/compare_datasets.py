"""
Comparative noise-model and dataset-characteristics research: old dataset
(semicon/train/train, 3200 pairs) vs. new dataset (semicon_train_data, 4785
pairs). Runs identical analysis methodology on both for a like-for-like
comparison. Standalone -- numpy/scipy/skimage/sklearn/matplotlib/opencv only.
"""
import glob
import hashlib
import json
import os
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "old": dict(
        gt_dir=os.path.join(ROOT, "semicon", "train", "train", "GT"),
        nlr_dir=os.path.join(ROOT, "semicon", "train", "train", "NoisyLR"),
        label="Old dataset (semicon/train/train)",
    ),
    "new": dict(
        gt_dir=os.path.join(ROOT, "semicon_train_data", "semicon_train_data", "GT"),
        nlr_dir=os.path.join(ROOT, "semicon_train_data", "semicon_train_data", "NoisyLR"),
        label="New dataset (semicon_train_data)",
    ),
}


def log(msg):
    print(f"[compare] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def area_downsample(gt):
    h, w = gt.shape
    return gt.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def bicubic_downsample(gt):
    return resize(gt, (gt.shape[0] // 2, gt.shape[1] // 2), order=3,
                  anti_aliasing=True, preserve_range=True).astype(np.float32)


def bicubic_upsample(lr, size):
    return resize(lr, size, order=3, anti_aliasing=False, preserve_range=True).astype(np.float32)


def radial_profile(mag2d):
    h, w = mag2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = min(cy, cx)
    tbin = np.bincount(r.ravel(), mag2d.ravel(), minlength=r_max + 1)[: r_max + 1]
    nr = np.bincount(r.ravel(), minlength=r_max + 1)[: r_max + 1]
    nr[nr == 0] = 1
    return tbin / nr


def degradation_dominance_score(degraded, block=16):
    """Pearson correlation of local mean vs local std over block tiles --
    high => speckle-like (std scales with signal), low => Gaussian-like
    (std roughly signal-independent). Same metric used by the project's own
    physics-informed synthetic degradation calibration."""
    h, w = degraded.shape[:2]
    means, stds = [], []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = degraded[y:y + block, x:x + block]
            means.append(float(patch.mean()))
            stds.append(float(patch.std()))
    means = np.array(means); stds = np.array(stds)
    if len(means) < 4 or means.std() < 1e-6 or stds.std() < 1e-6:
        return 0.0
    corr = np.corrcoef(means, stds)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def block_max_corr(img, block=32):
    h, w = img.shape
    nby, nbx = h // block, w // block
    if nby == 0 or nbx == 0:
        return 1.0
    blocks = img[:nby*block, :nbx*block].reshape(nby, block, nbx, block).transpose(0,2,1,3).reshape(-1, block, block)
    a = blocks[:, :, :-1].reshape(len(blocks), -1)
    b = blocks[:, :, 1:].reshape(len(blocks), -1)
    a2 = blocks[:, :-1, :].reshape(len(blocks), -1)
    b2 = blocks[:, 1:, :].reshape(len(blocks), -1)
    def rowcorr(X, Y):
        Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
        num = (Xc*Yc).sum(1)
        den = np.sqrt((Xc**2).sum(1) * (Yc**2).sum(1)) + 1e-9
        return num/den
    c = (rowcorr(a,b) + rowcorr(a2,b2)) / 2
    return float(np.max(c))


def analyze_dataset(key, cfg):
    gt_dir, nlr_dir, label = cfg["gt_dir"], cfg["nlr_dir"], cfg["label"]
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy")))
    log(f"[{key}] {label}: {len(files)} pairs found")
    t0 = time.time()

    n = len(files)
    gt_min, gt_max, lr_min, lr_max = np.inf, -np.inf, np.inf, -np.inf
    gt_means, gt_stds, lr_means, lr_stds = [], [], [], []
    frac_below0, frac_above1 = [], []
    resid_bins = np.linspace(-1, 1.5, 251)
    resid_hist = np.zeros(len(resid_bins) - 1)
    n_ibins = 10
    ibin_edges = np.linspace(0, 1, n_ibins + 1)
    resid_sq_sum = np.zeros(n_ibins)
    resid_count = np.zeros(n_ibins)
    mse_area_l, mse_bicubic_l = [], []
    radial_gt, radial_lr, radial_ds = [], [], []
    dom_scores = []
    gt_lap, lr_lap_up = [], []
    bicubic_psnr, bicubic_ssim = [], []
    block_max_corrs = []
    hashes = {}
    corrupt = []

    for f in files:
        try:
            gt = np.load(os.path.join(gt_dir, f)).astype(np.float32)
            lr = np.load(os.path.join(nlr_dir, f)).astype(np.float32)
        except Exception as e:
            corrupt.append((f, str(e)))
            continue

        gt_min = min(gt_min, gt.min()); gt_max = max(gt_max, gt.max())
        lr_min = min(lr_min, lr.min()); lr_max = max(lr_max, lr.max())
        gt_means.append(gt.mean()); gt_stds.append(gt.std())
        lr_means.append(lr.mean()); lr_stds.append(lr.std())
        frac_below0.append(float((lr < 0).mean()))
        frac_above1.append(float((lr > 1).mean()))

        gt_ds_area = area_downsample(gt)
        gt_ds_bic = bicubic_downsample(gt)
        resid = lr - gt_ds_area
        resid_hist += np.histogram(resid, bins=resid_bins)[0]

        bin_idx = np.clip(np.digitize(gt_ds_area, ibin_edges) - 1, 0, n_ibins - 1)
        for b in range(n_ibins):
            m = bin_idx == b
            if m.any():
                resid_sq_sum[b] += (resid[m] ** 2).sum()
                resid_count[b] += m.sum()

        mse_area_l.append(float(np.mean((lr - gt_ds_area) ** 2)))
        mse_bicubic_l.append(float(np.mean((lr - gt_ds_bic) ** 2)))

        dom_scores.append(degradation_dominance_score(lr))

        f_gt = np.abs(np.fft.fftshift(np.fft.fft2(gt))) ** 2
        f_lr = np.abs(np.fft.fftshift(np.fft.fft2(lr))) ** 2
        f_ds = np.abs(np.fft.fftshift(np.fft.fft2(gt_ds_area))) ** 2
        radial_gt.append(radial_profile(f_gt))
        radial_lr.append(radial_profile(f_lr))
        radial_ds.append(radial_profile(f_ds))

        lr_up = bicubic_upsample(lr, gt.shape)
        gt_lap.append(float(ndimage.laplace(gt).var()))
        lr_lap_up.append(float(ndimage.laplace(lr_up).var()))
        lr_up_c = np.clip(lr_up, 0, 1)
        bicubic_psnr.append(float(peak_signal_noise_ratio(gt, lr_up_c, data_range=1.0)))
        bicubic_ssim.append(float(structural_similarity(gt, lr_up_c, data_range=1.0)))

        block_max_corrs.append(block_max_corr(gt))
        h = hashlib.md5(gt.tobytes()).hexdigest()
        hashes.setdefault(h, []).append(f)

    gt_means = np.array(gt_means); gt_stds = np.array(gt_stds)
    lr_means = np.array(lr_means); lr_stds = np.array(lr_stds)
    frac_below0 = np.array(frac_below0); frac_above1 = np.array(frac_above1)
    dom_scores = np.array(dom_scores)
    bicubic_psnr = np.array(bicubic_psnr); bicubic_ssim = np.array(bicubic_ssim)
    gt_lap = np.array(gt_lap); lr_lap_up = np.array(lr_lap_up)
    block_max_corrs = np.array(block_max_corrs)
    resid_count_safe = np.where(resid_count == 0, 1, resid_count)
    variance_by_intensity = resid_sq_sum / resid_count_safe
    ibin_centers = (ibin_edges[:-1] + ibin_edges[1:]) / 2
    slope = float(np.polyfit(ibin_centers, variance_by_intensity, 1)[0])
    dupes = {h: v for h, v in hashes.items() if len(v) > 1}

    # tertile split like the project's own label_validation_set convention
    lo, hi = np.percentile(dom_scores, [33.33, 66.67])
    n_gauss = int((dom_scores <= lo).sum())
    n_mixed = int(((dom_scores > lo) & (dom_scores < hi)).sum())
    n_speckle = int((dom_scores >= hi).sum())

    max_len_gt = max(len(r) for r in radial_gt)
    max_len_lr = max(len(r) for r in radial_lr)
    def pad_mean(arrs, L):
        acc = np.zeros(L); cnt = np.zeros(L)
        for a in arrs:
            acc[: len(a)] += a; cnt[: len(a)] += 1
        return acc / np.maximum(cnt, 1)
    mean_radial_gt = pad_mean(radial_gt, max_len_gt)
    mean_radial_lr = pad_mean(radial_lr, max_len_lr)
    mean_radial_ds = pad_mean(radial_ds, max_len_lr)

    corrupted_files = [files[i] for i in np.where(block_max_corrs < 0.15)[0]] if len(files) else []

    result = {
        "key": key, "label": label, "n_pairs": n, "n_corrupt_load_errors": len(corrupt),
        "gt_shape_range": [int(gt.shape[0])], "resolution": f"{gt.shape[0]}x{gt.shape[0]} GT / {lr.shape[0]}x{lr.shape[0]} LR",
        "gt_global_range": [float(gt_min), float(gt_max)],
        "lr_global_range": [float(lr_min), float(lr_max)],
        "gt_mean": float(gt_means.mean()), "gt_std_mean": float(gt_stds.mean()),
        "lr_mean": float(lr_means.mean()), "lr_std_mean": float(lr_stds.mean()),
        "frac_lr_below0_mean_pct": float(frac_below0.mean() * 100),
        "frac_lr_above1_mean_pct": float(frac_above1.mean() * 100),
        "pct_images_any_below0": float((frac_below0 > 0).mean() * 100),
        "pct_images_any_above1": float((frac_above1 > 0).mean() * 100),
        "residual_variance_by_intensity_bin": variance_by_intensity.tolist(),
        "intensity_bin_centers": ibin_centers.tolist(),
        "signal_dependence_slope": slope,
        "mean_mse_vs_area_downsample": float(np.mean(mse_area_l)),
        "mean_mse_vs_bicubic_downsample": float(np.mean(mse_bicubic_l)),
        "closer_kernel": "area/box average" if np.mean(mse_area_l) < np.mean(mse_bicubic_l) else "bicubic",
        "degradation_dominance_score_mean": float(dom_scores.mean()),
        "degradation_dominance_score_std": float(dom_scores.std()),
        "n_gaussian_dominant": n_gauss, "n_mixed": n_mixed, "n_speckle_dominant": n_speckle,
        "gt_laplacian_var_mean": float(gt_lap.mean()),
        "lr_laplacian_var_mean_bicubic_up": float(lr_lap_up.mean()),
        "laplacian_ratio_lr_over_gt": float(lr_lap_up.mean() / gt_lap.mean()),
        "bicubic_psnr_mean": float(bicubic_psnr.mean()), "bicubic_psnr_std": float(bicubic_psnr.std()),
        "bicubic_ssim_mean": float(bicubic_ssim.mean()), "bicubic_ssim_std": float(bicubic_ssim.std()),
        "n_duplicate_gt_groups": len(dupes),
        "n_corrupted_gt_detected": len(corrupted_files),
        "corrupted_gt_files_sample": corrupted_files[:15],
        "elapsed_sec": time.time() - t0,
    }
    log(f"[{key}] done in {result['elapsed_sec']:.1f}s: bicubic PSNR={result['bicubic_psnr_mean']:.2f} "
        f"dom_score={result['degradation_dominance_score_mean']:.3f} slope={slope:.4f} "
        f"corrupted={result['n_corrupted_gt_detected']}")

    return result, dict(
        resid_hist=resid_hist, resid_bins=resid_bins,
        mean_radial_gt=mean_radial_gt, mean_radial_lr=mean_radial_lr, mean_radial_ds=mean_radial_ds,
        dom_scores=dom_scores, variance_by_intensity=variance_by_intensity, ibin_centers=ibin_centers,
        bicubic_psnr=bicubic_psnr, bicubic_ssim=bicubic_ssim,
    )


def main():
    results = {}
    arrays = {}
    for key, cfg in DATASETS.items():
        r, a = analyze_dataset(key, cfg)
        results[key] = r
        arrays[key] = a

    json.dump(results, open(os.path.join(HERE, "comparison_stats.json"), "w"), indent=2)
    log("wrote comparison_stats.json")

    # ---- comparison figures ----
    colors = {"old": "#c0392b", "new": "#2980b9"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in DATASETS:
        centers = (arrays[key]["resid_bins"][:-1] + arrays[key]["resid_bins"][1:]) / 2
        h = arrays[key]["resid_hist"]
        ax.plot(centers, h / h.sum(), label=DATASETS[key]["label"], color=colors[key])
    ax.set_title("Noise-only residual distribution (LR - area_downsample(GT))")
    ax.set_xlabel("residual value"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_residual_distribution_compare.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in DATASETS:
        ax.plot(arrays[key]["ibin_centers"], arrays[key]["variance_by_intensity"], marker="o",
                 label=DATASETS[key]["label"], color=colors[key])
    ax.set_title("Residual variance vs local GT intensity\n(flat=additive/Gaussian, rising=signal-dependent/speckle)")
    ax.set_xlabel("GT (downsampled) intensity"); ax.set_ylabel("residual variance"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_signal_dependence_compare.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in DATASETS:
        ax.hist(arrays[key]["dom_scores"], bins=40, alpha=0.55, label=DATASETS[key]["label"], color=colors[key], density=True)
    ax.set_title("Degradation-dominance score distribution\n(local-mean vs local-std correlation: high=speckle-like, low=Gaussian-like)")
    ax.set_xlabel("dominance score"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_degradation_dominance_compare.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key in DATASETS:
        axes[0].hist(arrays[key]["bicubic_psnr"], bins=40, alpha=0.55, label=DATASETS[key]["label"], color=colors[key], density=True)
        axes[1].hist(arrays[key]["bicubic_ssim"], bins=40, alpha=0.55, label=DATASETS[key]["label"], color=colors[key], density=True)
    axes[0].set_title("Bicubic-upsample PSNR distribution"); axes[0].legend()
    axes[1].set_title("Bicubic-upsample SSIM distribution"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_bicubic_baseline_compare.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in DATASETS:
        r_lr = arrays[key]["mean_radial_lr"]
        r_ds = arrays[key]["mean_radial_ds"]
        x = np.arange(len(r_lr)) / len(r_lr)
        ax.semilogy(x, r_lr, label=f"{DATASETS[key]['label']} - NoisyLR", color=colors[key])
        ax.semilogy(x, r_ds, label=f"{DATASETS[key]['label']} - clean area-ds(GT)", color=colors[key], ls="--", alpha=0.6)
    ax.set_title("Radially-averaged power spectrum (128px domain)")
    ax.set_xlabel("normalized spatial frequency"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "05_radial_spectrum_compare.png"), dpi=130)
    plt.close(fig)

    log("wrote 5 comparison figures")


if __name__ == "__main__":
    main()
