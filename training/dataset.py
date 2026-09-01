"""
Data pipeline for the restoration model.

Pairing: GT/<id>.npy <-> NoisyLR/<id>.npy, confirmed by the EDA (eda/eda_analysis.py,
section 1: 0 filename mismatches, 0 corrupt files across all 4785 pairs). GT is
256x256 float32 in [0,1]; NoisyLR is 128x128 float32, legitimately outside [0,1]
(EDA section 2: content-dependent overshoot/undershoot, not a fixed offset --
kept as-is here, never clipped before the model sees it).

Each real sample is optionally replaced (probability `synth_prob`) by a
synthetic degradation generated on-the-fly from its own GT via
degradation_sim.synth_degrade -- physics-informed (Gamma speckle + Gaussian +
randomized downsample kernel, shuffled order), ported unchanged from the
earlier project where its parameters were KS-test calibrated against real
data.

Returns, per sample:
  - nlr:     (1, H/2, W/2) raw low-res input, NOT clipped -- for models that
             operate at native LR resolution (e.g. SwinIR-lite)
  - bicubic: (1, H, W) bicubic-upsampled baseline -- fixed, non-learned,
             used as the residual-add baseline by both architectures
  - gt:      (1, H, W) ground truth, always in [0,1]
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from degradation_sim import synth_degrade

SR_SCALE = 2


def bicubic_upsample(x: torch.Tensor, size) -> torch.Tensor:
    """x: (C,H,W) tensor, any value range. Returns (C,H,W) bicubic-upsampled to `size`."""
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    return x.squeeze(0)


class RestorationDataset(Dataset):
    def __init__(self, files, gt_dir, nlr_dir, patch_size=64, augment=True,
                 synth_prob=0.3, seed=42):
        self.files = files
        self.gt_dir = gt_dir
        self.nlr_dir = nlr_dir
        self.patch_size = patch_size
        self.augment = augment
        self.synth_prob = synth_prob if augment else 0.0
        # per-worker independent RNG; re-seeded in worker_init_fn for DataLoader workers
        self.rng = np.random.default_rng(seed)

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        gt = np.load(os.path.join(self.gt_dir, fn)).astype(np.float32)

        if self.synth_prob > 0 and self.rng.random() < self.synth_prob:
            nlr = synth_degrade(gt, self.rng, scale=SR_SCALE)
        else:
            nlr = np.load(os.path.join(self.nlr_dir, fn)).astype(np.float32)

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        nlr_t = torch.from_numpy(nlr).unsqueeze(0)
        bicubic_t = bicubic_upsample(nlr_t, size=gt_t.shape[-2:])

        if self.augment:
            gt_t, bicubic_t, nlr_t = self._random_crop(gt_t, bicubic_t, nlr_t, self.patch_size)
            gt_t, bicubic_t, nlr_t = self._random_flip(gt_t, bicubic_t, nlr_t)

        return {"nlr": nlr_t, "bicubic": bicubic_t, "gt": gt_t, "filename": fn}

    @staticmethod
    def _random_crop(gt, bicubic, nlr, patch_size):
        _, h, w = gt.shape
        if h <= patch_size or w <= patch_size:
            return gt, bicubic, nlr
        top = torch.randint(0, h - patch_size + 1, (1,)).item()
        left = torch.randint(0, w - patch_size + 1, (1,)).item()
        gt_c = gt[:, top:top + patch_size, left:left + patch_size]
        bic_c = bicubic[:, top:top + patch_size, left:left + patch_size]
        nlr_c = nlr[:, top // 2:top // 2 + patch_size // 2, left // 2:left // 2 + patch_size // 2]
        return gt_c, bic_c, nlr_c

    @staticmethod
    def _random_flip(gt, bicubic, nlr):
        if torch.rand(1).item() < 0.5:
            gt, bicubic, nlr = gt.flip(-1), bicubic.flip(-1), nlr.flip(-1)
        if torch.rand(1).item() < 0.5:
            gt, bicubic, nlr = gt.flip(-2), bicubic.flip(-2), nlr.flip(-2)
        return gt, bicubic, nlr


def worker_init_fn(worker_id):
    """Re-seed each DataLoader worker's dataset copy so synthetic augmentation
    isn't identical across workers (each worker got a pickled copy of the same
    np.random.default_rng state otherwise)."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(42 + worker_id)


if __name__ == "__main__":
    import json
    HERE = os.path.dirname(os.path.abspath(__file__))
    manifest = json.load(open(os.path.join(HERE, "data_manifest.json")))
    ds = RestorationDataset(manifest["train_files"][:20], manifest["gt_dir"],
                             manifest["noisylr_dir"], patch_size=64, augment=True, synth_prob=0.3)
    sample = ds[0]
    print("nlr:", sample["nlr"].shape, f"range=[{sample['nlr'].min():.3f},{sample['nlr'].max():.3f}]")
    print("bicubic:", sample["bicubic"].shape, f"range=[{sample['bicubic'].min():.3f},{sample['bicubic'].max():.3f}]")
    print("gt:", sample["gt"].shape, f"range=[{sample['gt'].min():.3f},{sample['gt'].max():.3f}]")

    val_ds = RestorationDataset(manifest["val_files"][:5], manifest["gt_dir"],
                                 manifest["noisylr_dir"], augment=False)
    vs = val_ds[0]
    print("\nval (full-res, no augment) bicubic:", vs["bicubic"].shape, "gt:", vs["gt"].shape)
