"""
SwinIR-lightweight: a compact Swin-Transformer restoration network, the
second bake-off candidate against NAFNet-full (model_nafnet.py). New code --
no existing implementation to reuse -- built from the standard SwinIR design
(Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW
2021): shallow conv -> stacked Residual Swin Transformer Blocks (RSTB, each a
few window-attention Swin layers + a conv + a residual) -> upsample head.

Default config (60-dim embedding, 4 RSTBs x 6 layers, 8x8 windows, 6 heads)
mirrors the disclosed, working config from a public solution to this same
hackathon (see architecture_and_training_time_report.md section 2) as a
reasonable starting point -- not copied code, just a sane place to start.

Deliberately operates the transformer body at the *input* 128px resolution,
not on a pre-upsampled 256px image -- keeps token count (and therefore
window-attention cost) 4x lower than running at output resolution, which is
the whole point of this candidate: comparable quality to NAFNet-full at a
fraction of the params/inference cost (slide 15 scores inference time).
Upsampling to 256px happens only in the final PixelShuffle head, with the
same residual-over-bicubic-baseline design as the NAFNet candidate (residual
learning converges faster, and keeps both candidates' losses/training loop
identical -- only the model differs, per the bake-off's isolation goal).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """x: (B,H,W,C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B,H,W,C)"""
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2,Wh,Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2,N,N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N,N
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C), N = window_size*window_size"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        # Cached per (H, W, device): this mask depends only on the feature
        # map's resolution, not its content, so rebuilding it with Python-loop
        # slicing on every forward call is pure waste -- measured to cost
        # ~4x NAFNet's inference time before caching (see conversation), which
        # would have been a misleading basis for the architecture comparison.
        key = (H, W, str(device))
        cache = self.__dict__.setdefault("_mask_cache", {})
        if key in cache:
            return cache[key]
        img_mask = torch.zeros((1, H, W, 1), device=device)
        ws, ss = self.window_size, self.shift_size
        cnt = 0
        for h_slice in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w_slice in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        cache[key] = attn_mask
        return attn_mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        mask = self._attn_mask(H, W, x.device)
        attn_windows = self.attn(windows, mask=mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: `depth` SwinLayers (alternating
    shift=0/window_size//2) + a conv, wrapped in a residual connection."""
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinLayer(dim, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        shortcut = x
        for layer in self.layers:
            x = layer(x, H, W)
        B, L, C = x.shape
        x_conv = x.transpose(1, 2).view(B, C, H, W)
        x_conv = self.conv(x_conv).flatten(2).transpose(1, 2)
        return shortcut + x_conv


class SwinIRLite(nn.Module):
    def __init__(self, img_channel=1, embed_dim=60, depths=(6, 6, 6, 6), num_heads=6,
                 window_size=8, mlp_ratio=2.0, upscale=2):
        super().__init__()
        self.window_size = window_size
        self.upscale = upscale
        self.embed_dim = embed_dim

        self.conv_first = nn.Conv2d(img_channel, embed_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            RSTB(embed_dim, depth, num_heads, window_size, mlp_ratio) for depth in depths
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, img_channel, 3, 1, 1),
        )

    def _pad_to_window(self, x):
        _, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w

    def forward(self, bicubic, nlr):
        """nlr: (N,1,H/2,W/2) raw low-res input -- the transformer body runs at
        this (native LR) resolution. bicubic: (N,1,H,W) residual-add baseline,
        matching NAFNetRestoration's call signature for a uniform train/eval loop."""
        x, h0, w0 = self._pad_to_window(nlr)
        feat = self.conv_first(x)
        B, C, H, W = feat.shape
        seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for block in self.blocks:
            seq = block(seq, H, W)
        seq = self.norm(seq)

        feat2 = seq.transpose(1, 2).view(B, C, H, W)
        feat2 = self.conv_after_body(feat2) + feat
        feat2 = feat2[:, :, :h0, :w0]

        R = self.upsample(feat2)  # (N,1,2*h0,2*w0)
        out = torch.clamp(bicubic + R, 0.0, 1.0)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


LITE_CONFIG = dict(embed_dim=60, depths=(6, 6, 6, 6), num_heads=6, window_size=8, mlp_ratio=2.0)


def build_model(**kwargs):
    cfg = dict(LITE_CONFIG)
    cfg.update(kwargs)
    return SwinIRLite(img_channel=1, upscale=2, **cfg)


if __name__ == "__main__":
    m = build_model()
    n = m.num_params()
    print(f"SwinIR-lite: params={n} ({n/1e6:.3f}M)")
    nlr = torch.randn(2, 1, 128, 128)
    bicubic = torch.randn(2, 1, 256, 256)
    out = m(bicubic, nlr)
    print("  output:", out.shape, f"range=[{out.min():.3f},{out.max():.3f}]")
    out.sum().backward()
    print("  grad finite:", all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))
