"""
Config-driven training script for the restoration model. Covers both
bake-off candidates (--arch nafnet_full | swinir_lite) with one shared loop,
so the comparison isolates architecture only (same manifest, same loss, same
schedule) -- same discipline the earlier project's Phase A used correctly.

Key properties, per the plan:
  - Wall-clock training budget (--budget_minutes), progressive patch size
    (64px -> 128px) tied to elapsed time within the budget, AdamW + cosine LR.
  - AMP mixed precision, with SSIM/LPIPS/edge losses correctly forced to fp32
    internally (see losses.py) -- the earlier project's real NaN-gradient
    bug fix, carried forward.
  - Checkpointing: best-val-PSNR + periodic (every --ckpt_every_sec).
  - Real resume support: --resume <path> restores model, optimizer, and
    scaler state, plus step/epoch counters and elapsed-time-so-far, so a
    killed/restarted process continues rather than restarting -- the gap
    explicitly flagged in the earlier project's own README. (LR is a pure
    function of elapsed time, not scheduler state, so it needs no separate
    restore -- see set_lr_for_elapsed.)
  - Live status: a human-readable progress line every epoch, AND a
    status.json rewritten every epoch (state/epoch/elapsed/budget/loss/
    val metrics/last checkpoint path) so training status is queryable by
    just reading a file, not only by tailing stdout.

Usage:
    python train.py --arch nafnet_full --run_name bakeoff_nafnet --budget_minutes 25
    python train.py --arch swinir_lite --run_name bakeoff_swinir --budget_minutes 25
    python train.py --arch nafnet_full --run_name bakeoff_nafnet --budget_minutes 25 \
        --resume runs/bakeoff_nafnet/periodic.pth
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import RestorationDataset, worker_init_fn
from losses import LossConfig, CombinedLoss

# Python fully-buffers stdout when it's not a TTY (i.e. redirected to a log
# file, as for any long unattended run) -- print() output otherwise sits in
# an internal buffer and never reaches the log file until it fills or the
# process exits, even though the run itself is progressing normally
# (status.json, written via explicit open/write/close each epoch, is
# unaffected). Force line-buffering so the log is actually live.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def build_model_for_arch(arch, pretrained):
    if arch == "nafnet_full":
        if pretrained:
            from pretrained_loader import build_pretrained_nafnet
            model, report = build_pretrained_nafnet(pretrained=True)
            return model, report
        else:
            from model_nafnet import build_model, FULLSIZE_CONFIG
            return build_model(**FULLSIZE_CONFIG), {"pretrained": False}
    elif arch == "swinir_lite":
        from model_swinir_lite import build_model
        return build_model(), {"pretrained": False}
    else:
        raise ValueError(f"unknown arch {arch}")


def patch_size_for_elapsed(elapsed_sec, budget_sec):
    """First 60% of budget: 64px patches (fast iteration). Remaining 40%:
    128px (full native resolution, closer to eval conditions)."""
    return 64 if elapsed_sec < 0.6 * budget_sec else 128


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True,
                       persistent_workers=(num_workers > 0))


@torch.no_grad()
def evaluate_val(model, val_files, gt_dir, nlr_dir, device, max_samples=None):
    files = val_files if max_samples is None else val_files[:max_samples]
    psnrs, ssims = [], []
    ds = RestorationDataset(files, gt_dir, nlr_dir, augment=False)
    for i in range(len(ds)):
        s = ds[i]
        bicubic = s["bicubic"].unsqueeze(0).to(device)
        nlr = s["nlr"].unsqueeze(0).to(device)
        gt = s["gt"].numpy()[0]
        out = model(bicubic, nlr).float().cpu().numpy()[0, 0]
        psnrs.append(peak_signal_noise_ratio(gt, out, data_range=1.0))
        ssims.append(structural_similarity(gt, out, data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def write_status(status_path, **kwargs):
    tmp = status_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kwargs, f, indent=2)
    os.replace(tmp, status_path)  # atomic on POSIX -- never a half-written status.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["nafnet_full", "swinir_lite"], required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--budget_minutes", type=float, required=True)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patch64_batch", type=int, default=16)
    ap.add_argument("--patch128_batch", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--synth_prob", type=float, default=0.3)
    ap.add_argument("--val_max_samples", type=int, default=150)
    ap.add_argument("--val_every_sec", type=float, default=60)
    ap.add_argument("--ckpt_every_sec", type=float, default=180)
    ap.add_argument("--lpips", action="store_true", default=True)
    ap.add_argument("--no_lpips", dest="lpips", action="store_false")
    ap.add_argument("--lam_ssim", type=float, default=0.15)  # empirically validated over 0.84, see runs/ssimtest_*
    ap.add_argument("--lam_edge", type=float, default=0.08)
    ap.add_argument("--lam_lpips", type=float, default=0.08)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(HERE, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    status_path = os.path.join(run_dir, "status.json")
    history_path = os.path.join(run_dir, "history.json")

    manifest = json.load(open(os.path.join(HERE, "data_manifest.json")))
    gt_dir, nlr_dir = manifest["gt_dir"], manifest["noisylr_dir"]
    train_files, val_files = manifest["train_files"], manifest["val_files"]

    model, load_report = build_model_for_arch(args.arch, args.pretrained)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    budget_sec = args.budget_minutes * 60
    loss_cfg = LossConfig(charbonnier=True, ssim=True, lam_ssim=args.lam_ssim, edge=True, lam_edge=args.lam_edge,
                           lpips=args.lpips, lam_lpips=args.lam_lpips, lpips_warmup_steps=200)
    loss_fn = CombinedLoss(loss_cfg, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    min_lr_frac = 0.01  # cosine floor, as a fraction of args.lr -- never fully zero

    def set_lr_for_elapsed(e):
        """Manual cosine decay tied to wall-clock fraction of the budget, not
        step/epoch count -- correct under a progressive patch schedule where
        batches-per-epoch changes mid-run (a fixed-T_max torch scheduler
        stepped once/batch would silently decay on the wrong clock)."""
        frac = min(1.0, max(0.0, e / budget_sec))
        lr = args.lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + np.cos(np.pi * frac)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    start_elapsed = 0.0
    epoch = 0
    best_val_psnr = -1e9
    history = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        loss_fn.step_count = ckpt.get("loss_step_count", 0)
        start_elapsed = ckpt["elapsed_sec"]
        epoch = ckpt["epoch"]
        best_val_psnr = ckpt["best_val_psnr"]
        if os.path.exists(history_path):
            history = json.load(open(history_path))
        print(f"[train] resumed from {args.resume}: epoch={epoch} elapsed={start_elapsed:.1f}s "
              f"best_val_psnr={best_val_psnr:.3f}")

    print(f"[train] arch={args.arch} params={n_params/1e6:.3f}M device={device} "
          f"train={len(train_files)} val={len(val_files)} budget={budget_sec:.0f}s")
    if isinstance(load_report, dict) and "n_loaded" in load_report:
        print(f"[train] pretrained-loaded: {load_report['n_loaded']}, skipped: {load_report['n_skipped']}")

    t_start = time.time()
    last_val_time = -1e9
    last_ckpt_time = -1e9
    epoch_durations = []

    def elapsed():
        return start_elapsed + (time.time() - t_start)

    def save_checkpoint(path):
        torch.save({
            "arch": args.arch, "epoch": epoch, "elapsed_sec": elapsed(),
            "best_val_psnr": best_val_psnr, "loss_step_count": loss_fn.step_count,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": vars(args),
        }, path)

    current_patch = None
    loader = None
    train_ds = None
    train_loss = None  # guards the final status/print if budget expires before one epoch completes

    write_status(status_path, state="running", arch=args.arch, run_name=args.run_name,
                  epoch=epoch, elapsed_sec=elapsed(), budget_sec=budget_sec, n_params=n_params,
                  best_val_psnr=best_val_psnr, last_train_loss=None, last_val_psnr=None,
                  last_val_ssim=None, last_checkpoint=None)

    while elapsed() < budget_sec:
        patch = patch_size_for_elapsed(elapsed(), budget_sec)
        if patch != current_patch:
            current_patch = patch
            batch_size = args.patch64_batch if patch == 64 else args.patch128_batch
            train_ds = RestorationDataset(train_files, gt_dir, nlr_dir, patch_size=patch,
                                           augment=True, synth_prob=args.synth_prob)
            loader = make_loader(train_ds, batch_size, args.num_workers)
            print(f"[train] switching to patch={patch} batch={batch_size}")

        epoch_t0 = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        last_terms = {}

        for batch in loader:
            if elapsed() >= budget_sec:
                break
            cur_lr = set_lr_for_elapsed(elapsed())
            bicubic = batch["bicubic"].to(device, non_blocking=True)
            nlr = batch["nlr"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(bicubic, nlr)
                loss, terms = loss_fn(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_fn.step()

            loss_sum += loss.item()
            n_batches += 1
            last_terms = terms

        epoch += 1
        epoch_dur = time.time() - epoch_t0
        epoch_durations.append(epoch_dur)
        train_loss = loss_sum / max(1, n_batches)

        val_psnr, val_ssim = None, None
        if elapsed() - last_val_time >= args.val_every_sec:
            model.eval()
            val_psnr, val_ssim = evaluate_val(model, val_files, gt_dir, nlr_dir, device,
                                               max_samples=args.val_max_samples)
            last_val_time = elapsed()
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint(os.path.join(run_dir, "best.pth"))

        last_ckpt_path = None
        if elapsed() - last_ckpt_time >= args.ckpt_every_sec:
            last_ckpt_path = os.path.join(run_dir, "periodic.pth")
            save_checkpoint(last_ckpt_path)
            last_ckpt_time = elapsed()

        remaining = budget_sec - elapsed()
        avg_epoch = np.mean(epoch_durations[-10:])
        eta_more_epochs = int(remaining / avg_epoch) if avg_epoch > 0 else 0
        mm, ss = divmod(int(elapsed()), 60)
        bmm, bss = divmod(int(budget_sec), 60)
        val_str = f"val PSNR {val_psnr:.2f} SSIM {val_ssim:.3f}" if val_psnr is not None else "val -- (skipped this epoch)"
        print(f"[train] Epoch {epoch} | {mm:02d}:{ss:02d} elapsed / {bmm:02d}:{bss:02d} budget "
              f"({int(remaining)}s left, ~{eta_more_epochs} more epochs @ {avg_epoch:.1f}s/epoch) | "
              f"patch={current_patch} | loss {train_loss:.4f} | {val_str}")

        history.append({"epoch": epoch, "elapsed_sec": elapsed(), "patch_size": current_patch,
                         "train_loss": train_loss, "loss_terms": last_terms,
                         "val_psnr": val_psnr, "val_ssim": val_ssim, "lr": cur_lr})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        write_status(status_path, state="running", arch=args.arch, run_name=args.run_name,
                     epoch=epoch, elapsed_sec=elapsed(), budget_sec=budget_sec, n_params=n_params,
                     best_val_psnr=best_val_psnr, last_train_loss=train_loss,
                     last_val_psnr=val_psnr, last_val_ssim=val_ssim,
                     last_checkpoint=last_ckpt_path, avg_epoch_sec=float(avg_epoch),
                     eta_more_epochs=eta_more_epochs)

    # final checkpoint + status
    save_checkpoint(os.path.join(run_dir, "final.pth"))
    model.eval()
    final_val_psnr, final_val_ssim = evaluate_val(model, val_files, gt_dir, nlr_dir, device,
                                                    max_samples=args.val_max_samples)
    if final_val_psnr > best_val_psnr:
        best_val_psnr = final_val_psnr
        save_checkpoint(os.path.join(run_dir, "best.pth"))

    write_status(status_path, state="done", arch=args.arch, run_name=args.run_name,
                 epoch=epoch, elapsed_sec=elapsed(), budget_sec=budget_sec, n_params=n_params,
                 best_val_psnr=best_val_psnr, last_train_loss=train_loss,
                 last_val_psnr=final_val_psnr, last_val_ssim=final_val_ssim,
                 last_checkpoint=os.path.join(run_dir, "final.pth"))
    print(f"[train] DONE. epochs={epoch} final_val_psnr={final_val_psnr:.3f} "
          f"final_val_ssim={final_val_ssim:.3f} best_val_psnr={best_val_psnr:.3f}")


if __name__ == "__main__":
    main()
