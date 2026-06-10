"""
Decoder: Hierarchical Feature Aggregation Decoder (HFAD)

Architecture:
  - FPN-style lateral connections from backbone
  - CBAM attention at each scale
  - Cross-scale attention for deep→shallow feature propagation
  - Edge Refinement Module (ERM) for crisp boundaries

WHY U-Net style for COD?
  Skip connections preserve high-frequency spatial details lost during
  downsampling — exactly the fine boundary information needed to separate
  a camouflaged object from its near-identical background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CBAM, EfficientSelfAttention, CrossScaleAttention


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------

def conv_bn_relu(in_c: int, out_c: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# Feature Pyramid Neck: aligns all backbone stages to same channel count
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    Lateral 1×1 convolutions to project all backbone stages to `out_channels`.
    WHY: Different EfficientNet stages produce 24/32/56/160/448 channels.
    Normalizing to a single dimension makes fusion/attention trivial.
    """

    def __init__(self, in_channels: list, out_channels: int = 128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels
        ])
        self.smooths = nn.ModuleList([
            conv_bn_relu(out_channels, out_channels) for _ in in_channels
        ])

    def forward(self, features: list) -> list:
        # Project all to same channel count
        out = [smooth(lat(f))
               for lat, smooth, f in zip(self.laterals, self.smooths, features)]

        # Top-down pathway: propagate coarse semantics to fine scales
        for i in range(len(out) - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=out[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            out[i] = out[i] + up

        return out


# ---------------------------------------------------------------------------
# Decoder Block: upsampling + skip + attention
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_bn_relu(in_c + skip_c, out_c)
        self.res  = ResidualBlock(out_c)
        if use_attention:
            self.cbam = CBAM(out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        if self.use_attention:
            x = self.cbam(x)
        return x


# ---------------------------------------------------------------------------
# Edge Refinement Module (ERM)
# WHY: Camouflage boundaries are the hardest part. A dedicated edge branch
#      supervises the model on boundary prediction separately, forcing it to
#      learn crisp edge features independent of region classification.
# ---------------------------------------------------------------------------

class EdgeRefinementModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Multi-scale edge detection: different dilation rates capture edges at
        # different scales (fine cracks vs coarse silhouettes)
        self.edge_conv1 = nn.Conv2d(in_channels, 64, 3, padding=1,  dilation=1,  bias=False)
        self.edge_conv2 = nn.Conv2d(in_channels, 64, 3, padding=2,  dilation=2,  bias=False)
        self.edge_conv3 = nn.Conv2d(in_channels, 64, 3, padding=4,  dilation=4,  bias=False)

        self.fuse = conv_bn_relu(192, 64)
        self.edge_head = nn.Conv2d(64, 1, 1)

        # Gate: edge features refine the main feature map
        self.gate = nn.Sequential(
            conv_bn_relu(in_channels + 64, in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        e1 = F.relu(self.edge_conv1(x), inplace=True)
        e2 = F.relu(self.edge_conv2(x), inplace=True)
        e3 = F.relu(self.edge_conv3(x), inplace=True)

        edge_feat = self.fuse(torch.cat([e1, e2, e3], dim=1))
        edge_pred = self.edge_head(edge_feat)

        # Refine main feature map with edge gate
        gate = self.gate(torch.cat([x, edge_feat], dim=1))
        x_refined = x * gate + x

        return x_refined, edge_pred


# ---------------------------------------------------------------------------
# Full Decoder
# ---------------------------------------------------------------------------

class HFADecoder(nn.Module):
    """
    Hierarchical Feature Aggregation Decoder.

    Input:  5 FPN feature maps (all `fpn_channels` wide)
    Output: segmentation logits + edge logits
    """

    def __init__(self, fpn_channels: int = 128, decoder_channels: int = 64):
        super().__init__()
        C = fpn_channels
        D = decoder_channels

        # Self-attention on deepest feature (most semantic, most compact)
        self.global_attn = EfficientSelfAttention(C, num_heads=4, pool_size=8)

        # Decoder blocks (bottom-up)
        self.dec4 = DecoderBlock(C, C, D * 4)   # 1/16 → 1/8
        self.dec3 = DecoderBlock(D * 4, C, D * 2)  # 1/8  → 1/4
        self.dec2 = DecoderBlock(D * 2, C, D)       # 1/4  → 1/2
        self.dec1 = DecoderBlock(D, C, D)            # 1/2  → 1/1

        # Edge refinement at 1/4 scale (balance between detail and efficiency)
        self.erm = EdgeRefinementModule(D * 2)

        # Final segmentation head
        self.seg_head = nn.Sequential(
            conv_bn_relu(D, D),
            nn.Conv2d(D, 1, 1),
        )

    def forward(self, fpn_feats: list):
        # fpn_feats: [f1, f2, f3, f4, f5]  (fine → coarse)
        f1, f2, f3, f4, f5 = fpn_feats

        # Apply global attention on coarsest semantic features
        f5 = self.global_attn(f5)

        # Decode bottom-up
        d4 = self.dec4(f5, f4)
        d3 = self.dec3(d4, f3)

        # Edge refinement at 1/4 scale
        d3, edge_pred = self.erm(d3)

        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        # Upsample to full resolution (×2 since f1 is 1/2)
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)

        seg_pred = self.seg_head(d0)

        # Upsample edge pred to match seg pred
        edge_pred = F.interpolate(edge_pred, size=seg_pred.shape[-2:],
                                  mode="bilinear", align_corners=False)

        return seg_pred, edge_pred
