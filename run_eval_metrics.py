"""
Full evaluation: runs the model on all NoisyLR .npy files, compares outputs
against matched GT .npy files, and reports PSNR, SSIM, and LPIPS.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import build_model

NOISY_DIR = os.path.join(HERE, "data", "NoisyLR")
GT_DIR    = os.path.join(HERE, "data", "GT")
CHECKPOINT = os.path.join(HERE, "weights", "model_best.pth")

# ── device ──────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}", flush=True)
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}", flush=True)

# ── model ────────────────────────────────────────────────────────────────────
t0 = time.time()
model = build_model()
ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model = model.to(device).eval()
print(f"Model loaded ({model.num_params():,} params) in {time.time()-t0:.2f}s", flush=True)

# ── optional LPIPS ───────────────────────────────────────────────────────────
try:
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    use_lpips = True
    print("LPIPS : enabled (alex net)", flush=True)
except ImportError:
    use_lpips = False
    print("LPIPS : skipped (lpips package not found)", flush=True)

# ── file list ────────────────────────────────────────────────────────────────
noisy_files = sorted(f for f in os.listdir(NOISY_DIR) if f.endswith(".npy"))
gt_files    = set(os.listdir(GT_DIR))
paired = [f for f in noisy_files if f in gt_files]
missing = [f for f in noisy_files if f not in gt_files]

print(f"\nFiles  : {len(noisy_files)} NoisyLR | {len(gt_files)} GT | {len(paired)} paired", flush=True)
if missing:
    print(f"WARNING: {len(missing)} NoisyLR files have no matching GT — skipped", flush=True)

# ── inference + metrics ──────────────────────────────────────────────────────
def bicubic_upsample(x, scale=2):
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h*scale, w*scale), mode="bicubic", align_corners=False)

psnr_list, ssim_list, lpips_list, infer_ms_list = [], [], [], []
errors = []

print(f"\nRunning inference on {len(paired)} images...", flush=True)
t_total_start = time.time()

with torch.no_grad():
    for i, fname in enumerate(paired):
        try:
            # --- load inputs ---
            lr_np = np.load(os.path.join(NOISY_DIR, fname)).astype(np.float32)
            gt_np = np.load(os.path.join(GT_DIR,    fname)).astype(np.float32)
            if lr_np.ndim == 3: lr_np = lr_np[..., 0]
            if gt_np.ndim == 3: gt_np = gt_np[..., 0]

            # --- model forward ---
            x       = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
            bicubic = bicubic_upsample(x, scale=2)
            t_s = time.time()
            pred = model(bicubic)
            if device == "cuda": torch.cuda.synchronize()
            infer_ms_list.append((time.time() - t_s) * 1000)

            pred_np = pred.squeeze().float().cpu().numpy()   # (2H, 2W)

            # --- align GT to pred resolution ---
            # GT is expected at 2× (HR); if somehow it's still LR size, upsample it.
            if gt_np.shape != pred_np.shape:
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)
                gt_t = F.interpolate(gt_t, size=pred_np.shape, mode="bicubic", align_corners=False)
                gt_np = gt_t.squeeze().numpy()
            gt_np = np.clip(gt_np, 0.0, 1.0)

            # --- PSNR ---
            psnr_val = ski_psnr(gt_np, pred_np, data_range=1.0)
            psnr_list.append(psnr_val)

            # --- SSIM ---
            ssim_val = ski_ssim(gt_np, pred_np, data_range=1.0)
            ssim_list.append(ssim_val)

            # --- LPIPS ---
            if use_lpips:
                def to_lpips(a):
                    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)
                    return t.repeat(1, 3, 1, 1) * 2 - 1   # [0,1] → [-1,1], RGB-like
                lp = lpips_fn(to_lpips(pred_np), to_lpips(gt_np)).item()
                lpips_list.append(lp)

            if (i+1) % 50 == 0 or (i+1) == len(paired):
                print(f"  [{i+1:>3}/{len(paired)}]  PSNR={np.mean(psnr_list):.4f}  SSIM={np.mean(ssim_list):.4f}"
                      + (f"  LPIPS={np.mean(lpips_list):.4f}" if use_lpips else "")
                      + f"  ({np.mean(infer_ms_list):.1f}ms/img)", flush=True)

        except Exception as e:
            errors.append((fname, str(e)))
            print(f"  ERROR on {fname}: {e}", flush=True)

t_total = time.time() - t_total_start

# ── results ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Images evaluated   : {len(psnr_list)}")
print(f"  Errors / skipped   : {len(errors)}")
print()
print(f"  PSNR  mean  : {np.mean(psnr_list):.4f} dB")
print(f"  PSNR  std   : {np.std(psnr_list):.4f} dB")
print(f"  PSNR  min   : {np.min(psnr_list):.4f} dB")
print(f"  PSNR  max   : {np.max(psnr_list):.4f} dB")
print()
print(f"  SSIM  mean  : {np.mean(ssim_list):.4f}")
print(f"  SSIM  std   : {np.std(ssim_list):.4f}")
print(f"  SSIM  min   : {np.min(ssim_list):.4f}")
print(f"  SSIM  max   : {np.max(ssim_list):.4f}")
if use_lpips and lpips_list:
    print()
    print(f"  LPIPS mean  : {np.mean(lpips_list):.4f}")
    print(f"  LPIPS std   : {np.std(lpips_list):.4f}")
    print(f"  LPIPS min   : {np.min(lpips_list):.4f}")
    print(f"  LPIPS max   : {np.max(lpips_list):.4f}")
print()
print(f"  Infer ms/img: {np.mean(infer_ms_list):.2f}ms  (std {np.std(infer_ms_list):.2f}ms)")
print(f"  Total wall  : {t_total:.1f}s  (model load + all inference + metrics)")
print("=" * 60)

# ── save JSON ─────────────────────────────────────────────────────────────────
results = {
    "n_evaluated": len(psnr_list),
    "n_errors": len(errors),
    "psnr_mean": float(np.mean(psnr_list)),
    "psnr_std":  float(np.std(psnr_list)),
    "psnr_min":  float(np.min(psnr_list)),
    "psnr_max":  float(np.max(psnr_list)),
    "ssim_mean": float(np.mean(ssim_list)),
    "ssim_std":  float(np.std(ssim_list)),
    "ssim_min":  float(np.min(ssim_list)),
    "ssim_max":  float(np.max(ssim_list)),
    "infer_ms_mean": float(np.mean(infer_ms_list)),
    "infer_ms_std":  float(np.std(infer_ms_list)),
    "total_wall_sec": t_total,
    "device": device,
}
if use_lpips and lpips_list:
    results.update({
        "lpips_mean": float(np.mean(lpips_list)),
        "lpips_std":  float(np.std(lpips_list)),
        "lpips_min":  float(np.min(lpips_list)),
        "lpips_max":  float(np.max(lpips_list)),
    })

out_json = os.path.join(HERE, "eval_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_json}")
