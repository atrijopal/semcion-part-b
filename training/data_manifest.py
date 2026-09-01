"""
REFERENCE ONLY -- included to document how data_manifest.json (already
provided in this folder, ready to use) was derived. Not meant to be re-run
from here: it depends on this project's original EDA working directory
(eda/outputs/split_manifest.json), which isn't part of this submission.
data_manifest.json's train/val split is the actual, final, resolved output
of this script's logic -- train.py reads that file directly and does not
invoke this script.

Builds the fixed train/val manifest for the restoration model: starts from
a cluster-stratified train/val split (see the EDA report) and removes the
30 files identified as genuinely-corrupted GT (corrupted_gt_exclusion.json
-- validated via max lag-1 spatial autocorrelation over 32x32 blocks,
cross-checked visually) from wherever they landed in that split.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # "SEMICON PART B"

EDA_SPLIT_PATH = os.path.join(ROOT, "eda", "outputs", "split_manifest.json")
EXCLUSION_PATH = os.path.join(HERE, "corrupted_gt_exclusion.json")
DATA_ROOT = os.path.join(ROOT, "semicon_train_data", "semicon_train_data")
GT_DIR = os.path.join(DATA_ROOT, "GT")
NOISYLR_DIR = os.path.join(DATA_ROOT, "NoisyLR")
OUT_PATH = os.path.join(HERE, "data_manifest.json")


def build_manifest():
    with open(EDA_SPLIT_PATH) as f:
        eda_split = json.load(f)
    with open(EXCLUSION_PATH) as f:
        exclusion = json.load(f)

    excluded = set(exclusion["flagged_files"])
    train_files = [f for f in eda_split["train_files"] if f not in excluded]
    val_files = [f for f in eda_split["val_files"] if f not in excluded]

    n_removed_train = len(eda_split["train_files"]) - len(train_files)
    n_removed_val = len(eda_split["val_files"]) - len(val_files)

    manifest = {
        "seed": eda_split["seed"],
        "gt_dir": GT_DIR,
        "noisylr_dir": NOISYLR_DIR,
        "source_split": EDA_SPLIT_PATH,
        "source_split_strategy": eda_split["strategy"],
        "exclusion_source": EXCLUSION_PATH,
        "exclusion_method": exclusion["method"],
        "n_excluded_total": exclusion["n_flagged"],
        "n_excluded_from_train": n_removed_train,
        "n_excluded_from_val": n_removed_val,
        "n_train": len(train_files),
        "n_val": len(val_files),
        "train_files": train_files,
        "val_files": val_files,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    m = build_manifest()
    print(f"train={m['n_train']} (removed {m['n_excluded_from_train']} corrupted), "
          f"val={m['n_val']} (removed {m['n_excluded_from_val']} corrupted)")
    print(f"wrote {OUT_PATH}")
