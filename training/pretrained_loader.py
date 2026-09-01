"""
NAFNet-SIDD pretrained checkpoint loader + grayscale channel adaptation.
Ported from the earlier project's pretrained_loader.py (Experiment A1 clearly
showed pretrained init winning) -- the official checkpoint is bundled locally
at pretrained/NAFNet-SIDD-width32.pth so this reproduces without any external
clone/download (source: megvii-research/NAFNet released checkpoints).

Channel adaptation for grayscale: average the pretrained intro conv's 3
input-channel weights into 1 (dim=1 mean), and average the ending conv's 3
output-channel weights into 1 (dim=0 mean) -- rather than discarding and
reinitializing those layers.

Bias handling: this codebase's convention is bias-free convolutions (Mohan
et al. / Restormer -- improves generalization across noise levels not seen
in training, relevant to the OOD test set). The pretrained checkpoint was
trained WITH biases; bias tensors are dropped from the loaded state dict and
every conv's bias is stripped post-load -- an accepted, documented tradeoff.
"""
import os

import torch
import torch.nn as nn

from model_nafnet import build_model, strip_conv_bias, FULLSIZE_CONFIG

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "NAFNet-SIDD-width32.pth")


def build_pretrained_nafnet(pretrained: bool = True):
    """Builds the full NAFNet-SIDD-sized additive-only backbone, either
    pretrained-and-adapted or freshly random-initialized. Returns
    (model, load_report) where load_report lists which top-level modules
    were loaded from the checkpoint vs. left at their (post-build) init."""
    model = build_model(**FULLSIZE_CONFIG)

    if not pretrained:
        report = {"loaded_from_checkpoint": [], "reinitialized": ["ALL (random init run)"]}
        return model, report

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {CKPT_PATH}. Expected it already downloaded (Phase 1)."
        )

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("params", ckpt)

    own_sd = model.state_dict()
    loaded, skipped, adapted = [], [], []

    for key, tensor in sd.items():
        if key.endswith(".bias"):
            continue  # bias-free convention -- drop pretrained biases

        # map their top-level names to ours
        own_key = key
        if key.startswith("ending."):
            own_key = key.replace("ending.", "head_add.")

        if own_key not in own_sd:
            skipped.append(key)
            continue

        own_shape = own_sd[own_key].shape
        if tensor.shape == own_shape:
            own_sd[own_key] = tensor
            loaded.append(own_key)
        elif own_key == "intro.weight" and tensor.shape[1] == 3 and own_shape[1] == 1:
            own_sd[own_key] = tensor.mean(dim=1, keepdim=True)
            adapted.append(f"{own_key} (avg 3->1 input channels)")
        elif own_key == "head_add.weight" and tensor.shape[0] == 3 and own_shape[0] == 1:
            own_sd[own_key] = tensor.mean(dim=0, keepdim=True)
            adapted.append(f"{own_key} (avg 3->1 output channels)")
        else:
            skipped.append(f"{key} (shape mismatch {tuple(tensor.shape)} vs {tuple(own_shape)})")

    model.load_state_dict(own_sd)
    strip_conv_bias(model)  # re-strip in case load_state_dict re-created bias buffers

    report = {
        "checkpoint": CKPT_PATH,
        "loaded_from_checkpoint_exact": loaded,
        "loaded_from_checkpoint_adapted": adapted,
        "skipped": skipped,
        "n_loaded": len(loaded) + len(adapted),
        "n_skipped": len(skipped),
    }
    return model, report


if __name__ == "__main__":
    for pretrained in (False, True):
        model, report = build_pretrained_nafnet(pretrained=pretrained)
        n = model.num_params()
        print(f"\npretrained={pretrained}: params={n} ({n/1e6:.2f}M)")
        if pretrained:
            print(f"  loaded exact: {len(report['loaded_from_checkpoint_exact'])}")
            print(f"  loaded adapted: {report['loaded_from_checkpoint_adapted']}")
            print(f"  skipped: {report['skipped'][:5]}{'...' if len(report['skipped']) > 5 else ''} "
                  f"(total {report['n_skipped']})")
        x = torch.randn(1, 1, 96, 96)
        out = model(x)
        print("  output:", out.shape, f"range=[{out.min():.3f},{out.max():.3f}]")
