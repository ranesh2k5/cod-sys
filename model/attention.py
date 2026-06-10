"""
Attention modules for Camouflaged Object Detection.

WHY attention for COD?
  - Camouflaged objects are "hidden" in global context. A CNN sees only local
    patches and cannot reason: "this texture patch is suspicious BECAUSE the
    surrounding region is forest floor."
  - Self-attention lets every pixel attend to every other pixel, learning
    which regions globally contradict each other (object vs background).
  - Channel attention suppresses uninformative feature maps and amplifies
    texture-discriminative channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Channel Attention (SE-style)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Learns WHICH feature channels matter — ignores channels that encode
    background-like information, amplifies object-boundary channels.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx  = self.fc(self.max_pool(x).view(B, C))
        w   = self.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    Spatial attention highlights WHERE in the image to focus.
    For COD: emphasizes edge/boundary regions where camouflage breaks down.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        pool  = torch.cat([avg, mx], dim=1)
        w     = self.sigmoid(self.conv(pool))
        return x * w


# ---------------------------------------------------------------------------
# CBAM: Combined Channel + Spatial
# ---------------------------------------------------------------------------

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# Efficient Self-Attention (Axial / Pooling-based)
# ---------------------------------------------------------------------------

class EfficientSelfAttention(nn.Module):
    """
    Pooling-based self-attention for spatial feature maps.
    Full O(N²) attention on 32×32 maps is fine, but we use pooled keys/values
    for larger maps to keep memory tractable.

    WHY self-attention for COD specifically?
      Camouflaged objects rely on SIMILARITY with surroundings. Self-attention
      computes pairwise similarity across the whole image — this makes it
      naturally suited to finding regions that are "too similar" to background.
    """

    def __init__(self, dim: int, num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.pool_size  = pool_size

        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Downsample keys/values to reduce memory
        x_pool = F.adaptive_avg_pool2d(x, self.pool_size)

        q = self.q(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.k(x_pool).reshape(B, self.num_heads, self.head_dim, self.pool_size ** 2)
        v = self.v(x_pool).reshape(B, self.num_heads, self.head_dim, self.pool_size ** 2)

        q = q.permute(0, 1, 3, 2)  # B, heads, HW, head_dim
        k = k.permute(0, 1, 3, 2)  # B, heads, pool², head_dim
        v = v.permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                        # B, heads, HW, head_dim
        out = out.permute(0, 1, 3, 2)          # B, heads, head_dim, HW
        out = out.reshape(B, C, H, W)

        out = self.proj(out) + residual

        # Apply layernorm channel-wise
        out_ln = out.permute(0, 2, 3, 1)        # B H W C
        out_ln = self.norm(out_ln)
        out = out_ln.permute(0, 3, 1, 2)        # B C H W

        return out


# ---------------------------------------------------------------------------
# Cross-Scale Attention (for feature fusion in decoder)
# ---------------------------------------------------------------------------

class CrossScaleAttention(nn.Module):
    """
    Attends a coarse feature map (lower resolution) to a fine feature map.
    Used in decoder to propagate global context to high-resolution features.
    """

    def __init__(self, fine_dim: int, coarse_dim: int, out_dim: int):
        super().__init__()
        self.q_proj = nn.Conv2d(fine_dim, out_dim, 1)
        self.k_proj = nn.Conv2d(coarse_dim, out_dim, 1)
        self.v_proj = nn.Conv2d(coarse_dim, out_dim, 1)
        self.out_proj = nn.Conv2d(out_dim, out_dim, 1)
        self.scale = out_dim ** -0.5

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        B, _, H, W = fine.shape

        # Upsample coarse to match fine resolution
        coarse_up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)

        q = self.q_proj(fine).reshape(B, -1, H * W).permute(0, 2, 1)   # B, HW, C
        k = self.k_proj(coarse_up).reshape(B, -1, H * W).permute(0, 2, 1)
        v = self.v_proj(coarse_up).reshape(B, -1, H * W).permute(0, 2, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v                              # B, HW, C
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        return self.out_proj(out)
