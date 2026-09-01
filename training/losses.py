"""
Losses module: Charbonnier, SSIM (forced fp32 under AMP autocast, per the
earlier project's Phase 1 NaN-gradient bug fix -- ported unchanged, not
rediscovered), small-weight TV, LPIPS (via the `lpips` pip package, AlexNet
backbone, with a linear warmup so its noisy early-training gradients don't
destabilize the pixel/structure terms), and a new edge/gradient (Laplacian)
term targeting the EDA's sharpest finding: NoisyLR's Laplacian variance is
~0.35x GT's (real high-frequency detail loss, distinct from noise-inflated
Sobel energy) -- see architecture_and_training_time_report.md section 3.
Each term individually toggleable via LossConfig; total is their weighted sum.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


_LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def laplacian(x):
    """x: (N,1,H,W). Returns the Laplacian response, same spatial size (replicate-padded)."""
    k = _LAPLACIAN_KERNEL.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_pad, k)


def edge_loss(pred, target, eps=1e-3):
    return charbonnier_loss(laplacian(pred), laplacian(target), eps=eps)


def _gaussian_window(window_size, sigma, channel, device, dtype):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device=device, dtype=dtype)


def ssim_torch(img1, img2, window_size=11, data_range=1.0):
    """Forced fp32 even under AMP autocast -- fp16 produces NaN gradients in
    the squared/divided SSIM terms (Phase 1 finding)."""
    with torch.cuda.amp.autocast(enabled=False):
        img1 = img1.float()
        img2 = img2.float()
        channel = img1.size(1)
        window = _gaussian_window(window_size, 1.5, channel, img1.device, img1.dtype)
        pad = window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
        C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def tv_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class LPIPSLoss(nn.Module):
    """Wraps the `lpips` package. Expects 1-channel [0,1] images; LPIPS wants
    3-channel [-1,1] -- replicate the grayscale channel to 3 and rescale.
    Always runs in fp32 (small net, cheap, and avoids AMP dtype mismatches
    with its internal, separately-loaded AlexNet weights)."""
    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net=net).to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            pred3 = pred_f.repeat(1, 3, 1, 1) * 2 - 1
            target3 = target_f.repeat(1, 3, 1, 1) * 2 - 1
            return self.net(pred3, target3).mean()


class LossConfig:
    def __init__(self, charbonnier=True, ssim=True, lam_ssim=0.84,
                 tv=False, lam_tv=0.02, lpips=False, lam_lpips=0.08,
                 lpips_warmup_steps=0, edge=True, lam_edge=0.08):
        self.charbonnier = charbonnier
        self.ssim = ssim
        self.lam_ssim = lam_ssim
        self.tv = tv
        self.lam_tv = lam_tv
        self.lpips = lpips
        self.lam_lpips = lam_lpips
        self.lpips_warmup_steps = lpips_warmup_steps
        self.edge = edge
        self.lam_edge = lam_edge

    def as_dict(self):
        return dict(charbonnier=self.charbonnier, ssim=self.ssim, lam_ssim=self.lam_ssim,
                    tv=self.tv, lam_tv=self.lam_tv, lpips=self.lpips, lam_lpips=self.lam_lpips,
                    lpips_warmup_steps=self.lpips_warmup_steps, edge=self.edge, lam_edge=self.lam_edge)


class CombinedLoss(nn.Module):
    def __init__(self, config: LossConfig, device="cuda"):
        super().__init__()
        self.config = config
        self.lpips_fn = LPIPSLoss(device=device) if config.lpips else None
        self.step_count = 0

    def step(self):
        """Call once per training iteration -- drives the LPIPS warmup ramp."""
        self.step_count += 1

    def _lpips_weight(self):
        w = self.config.lam_lpips
        n = self.config.lpips_warmup_steps
        if n <= 0:
            return w
        return w * min(1.0, self.step_count / n)

    def forward(self, pred, target):
        total = torch.zeros((), device=pred.device, dtype=torch.float32)
        terms = {}
        if self.config.charbonnier:
            ch = charbonnier_loss(pred, target)
            total = total + ch
            terms["charbonnier"] = ch.item()
        if self.config.ssim:
            s = ssim_torch(pred, target)
            total = total + self.config.lam_ssim * (1.0 - s)
            terms["ssim"] = s.item()
        if self.config.tv:
            tv = tv_loss(pred)
            total = total + self.config.lam_tv * tv
            terms["tv"] = tv.item()
        if self.config.edge:
            e = edge_loss(pred, target)
            total = total + self.config.lam_edge * e
            terms["edge"] = e.item()
        if self.config.lpips:
            w = self._lpips_weight()
            lp = self.lpips_fn(pred, target)
            total = total + w * lp
            terms["lpips"] = lp.item()
            terms["lpips_weight"] = w
        return total, terms


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LossConfig(charbonnier=True, ssim=True, tv=True, lpips=False)
    loss_fn = CombinedLoss(cfg, device=device)
    x = torch.rand(2, 1, 64, 64, device=device, requires_grad=True)
    y = torch.rand(2, 1, 64, 64, device=device)
    total, terms = loss_fn(x, y)
    print("total:", total.item(), "terms:", terms)
    total.backward()
    print("grad finite:", torch.isfinite(x.grad).all().item())
