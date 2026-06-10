"""
CODNet: Camouflaged Object Detection Network

Full pipeline:
    Input Image
        ↓
    EfficientNet-B4 Backbone  (multi-scale CNN features)
        ↓
    FPN Neck                  (normalise channels, top-down fusion)
        ↓
    EfficientSelfAttention    (global context on deepest features)
        ↓
    HFA Decoder               (hierarchical bottom-up decoding + CBAM)
        ↓
    Edge Refinement Module    (boundary-aware gating)
        ↓
    Seg head + Edge head      (binary segmentation + edge prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EfficientNetBackbone
from .decoder import FPNNeck, HFADecoder


class CODNet(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        fpn_channels: int = 128,
        decoder_channels: int = 64,
    ):
        super().__init__()

        self.backbone = EfficientNetBackbone(pretrained=pretrained_backbone)
        self.fpn      = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.decoder  = HFADecoder(fpn_channels, decoder_channels)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]  — H, W must be divisible by 32
        Returns:
            seg_logits:  [B, 1, H, W]
            edge_logits: [B, 1, H, W]
        """
        # Backbone
        feats = self.backbone(x)            # tuple of 5 tensors

        # FPN neck
        fpn_feats = self.fpn(list(feats))   # list of 5 tensors, all C=fpn_channels

        # Decode
        seg_logits, edge_logits = self.decoder(fpn_feats)

        # Ensure output matches input resolution
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits  = F.interpolate(seg_logits,  size=x.shape[-2:], mode="bilinear", align_corners=False)
            edge_logits = F.interpolate(edge_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return seg_logits, edge_logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference-only. Returns binary mask [B, 1, H, W]."""
        self.eval()
        with torch.no_grad():
            seg_logits, _ = self.forward(x)
            return (seg_logits.sigmoid() > threshold).float()


def build_model(pretrained: bool = True) -> CODNet:
    return CODNet(pretrained_backbone=pretrained)
