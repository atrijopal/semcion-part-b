"""
Loads a checkpoint, computes overall PSNR/SSIM/LPIPS on the manifest's held-
out val split, and measures inference time/image (GPU-warmed-up first --
the earlier project's documented lesson: cuDNN's one-time kernel-selection
overhead on the very first call otherwise dominates the average).

Usage:
    python evaluate.py --checkpoint runs/bakeoff_nafnet/best.pth --arch nafnet_full
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import RestorationDataset

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(arch, checkpoint_path, device):
    if arch == "nafnet_full":
        from model_nafnet import build_model, FULLSIZE_CONFIG
        model = build_model(**FULLSIZE_CONFIG)
    elif arch == "swinir_lite":
        from model_swinir_lite import build_model
        model = build_model()
    else:
        raise ValueError(arch)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def run_eval(model, val_files, gt_dir, nlr_dir, device, lpips_fn=None, n_warmup=5):
    ds = RestorationDataset(val_files, gt_dir, nlr_dir, augment=False)

    # GPU warmup
    if device == "cuda" and len(ds) > 0:
        s = ds[0]
        bicubic = s["bicubic"].unsqueeze(0).to(device)
        nlr = s["nlr"].unsqueeze(0).to(device)
        for _ in range(n_warmup):
            model(bicubic, nlr)
        torch.cuda.synchronize()

    psnrs, ssims, lpipses, infer_ms = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        bicubic = s["bicubic"].unsqueeze(0).to(device)
        nlr = s["nlr"].unsqueeze(0).to(device)
        gt_t = s["gt"].unsqueeze(0).to(device)
        gt = s["gt"].numpy()[0]

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(bicubic, nlr)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms.append((time.time() - t0) * 1000)

        out_np = out.float().cpu().numpy()[0, 0]
        psnrs.append(peak_signal_noise_ratio(gt, out_np, data_range=1.0))
        ssims.append(structural_similarity(gt, out_np, data_range=1.0))
        if lpips_fn is not None:
            with torch.no_grad():
                lp = lpips_fn(out.float(), gt_t.float()).item()
            lpipses.append(lp)

    result = {
        "n_samples": len(ds),
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "infer_ms_mean": float(np.mean(infer_ms)), "infer_ms_std": float(np.std(infer_ms)),
    }
    if lpipses:
        result["lpips_mean"] = float(np.mean(lpipses))
        result["lpips_std"] = float(np.std(lpipses))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", choices=["nafnet_full", "swinir_lite"], required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--lpips", action="store_true", default=True)
    ap.add_argument("--no_lpips", dest="lpips", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.load(open(os.path.join(HERE, "data_manifest.json")))
    val_files = manifest["val_files"]
    if args.max_samples:
        val_files = val_files[:args.max_samples]

    model, ckpt = load_model(args.arch, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())

    lpips_fn = None
    if args.lpips:
        from losses import LPIPSLoss
        lpips_fn = LPIPSLoss(device=device)

    result = run_eval(model, val_files, manifest["gt_dir"], manifest["noisylr_dir"], device, lpips_fn)
    result["arch"] = args.arch
    result["checkpoint"] = args.checkpoint
    result["n_params"] = n_params
    result["ckpt_epoch"] = ckpt.get("epoch")
    result["ckpt_elapsed_sec"] = ckpt.get("elapsed_sec")

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
