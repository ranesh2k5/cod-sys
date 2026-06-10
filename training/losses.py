from typing import Tuple, Dict
"""
Loss Functions for Camouflaged Object Detection

WHY each loss:
  1. BCE (Binary Cross-Entropy)
     Standard pixel-wise classification. Provides smooth gradients everywhere.

  2. Dice Loss
     Directly optimises the overlap metric (IoU-like).
     CRITICAL for COD: handles class imbalance (background >> foreground pixels)
     by normalising against total positive predictions + total GT positives.

  3. Weighted BCE (structure loss)
     Down-weights easy background pixels; up-weights object/boundary pixels.
     Common in SOD/COD literature — forces focus on hard regions.

  4. Edge / Boundary Loss
     Supervised on morphological edges of the GT mask.
     Directly penalises blurry boundaries — the #1 failure mode in COD.

  5. IoU Loss
     Differentiable IoU approximation. Complementary to Dice (they differ
     on false negatives/positives weighting).

Total loss = λ_seg * (BCE + Dice + IoU) + λ_edge * (BCE_edge + Dice_edge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid()
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


# ---------------------------------------------------------------------------
# IoU Loss (soft)
# ---------------------------------------------------------------------------

class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = logits.sigmoid()
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)

        intersection = (p_flat * t_flat).sum()
        union        = p_flat.sum() + t_flat.sum() - intersection
        iou          = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


# ---------------------------------------------------------------------------
# Structure-Weighted BCE
# WHY: weights high near edges/object pixels to penalise boundary errors more
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """
    Pixel-wise BCE weighted by local structure.
    Pixels near edges/boundaries get higher weight.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute structural weight map
        with torch.no_grad():
            # Gradient magnitude of target mask = boundary weight
            # Use simple Laplacian kernel
            kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=targets.device
            ).view(1, 1, 3, 3)

            edge_weight = torch.abs(
                F.conv2d(targets.float(), kernel, padding=1)
            ).clamp(0, 1)

            # Base weight 1, boundary weight up to 5
            weight = 1.0 + 4.0 * edge_weight

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce * weight).mean()


# ---------------------------------------------------------------------------
# Combined Segmentation Loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    Combined loss = Weighted-BCE + Dice + IoU

    This triplet is the standard in COD/SOD literature because:
    - WeightedBCE:  handles boundary focus
    - Dice:         handles class imbalance
    - IoU:          directly optimises the evaluation metric
    """

    def __init__(
        self,
        bce_weight:  float = 1.0,
        dice_weight: float = 1.0,
        iou_weight:  float = 1.0,
    ):
        super().__init__()
        self.w_bce  = bce_weight
        self.w_dice = dice_weight
        self.w_iou  = iou_weight

        self.wbce = WeightedBCELoss()
        self.dice = DiceLoss()
        self.iou  = IoULoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        l_bce  = self.wbce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_iou  = self.iou(logits, targets)

        total = (
            self.w_bce  * l_bce  +
            self.w_dice * l_dice +
            self.w_iou  * l_iou
        )
        return total, {"bce": l_bce.item(), "dice": l_dice.item(), "iou": l_iou.item()}


# ---------------------------------------------------------------------------
# Total COD Loss (seg + edge branch)
# ---------------------------------------------------------------------------

class CODLoss(nn.Module):
    """
    Final loss combining segmentation and edge supervision.

    Args:
        seg_weight:  weight for main segmentation loss
        edge_weight: weight for edge branch loss (lower — auxiliary task)
    """

    def __init__(self, seg_weight: float = 1.0, edge_weight: float = 0.4):
        super().__init__()
        self.seg_weight  = seg_weight
        self.edge_weight = edge_weight

        self.seg_loss  = SegmentationLoss()
        self.edge_dice = DiceLoss()
        self.edge_bce  = nn.BCEWithLogitsLoss()

    def forward(
        self,
        seg_logits:  torch.Tensor,
        edge_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        edge_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # Segmentation loss
        seg_total, seg_components = self.seg_loss(seg_logits, seg_targets)

        # Edge loss
        edge_bce  = self.edge_bce(edge_logits, edge_targets)
        edge_dice = self.edge_dice(edge_logits, edge_targets)
        edge_total = edge_bce + edge_dice

        total = self.seg_weight * seg_total + self.edge_weight * edge_total

        components = {
            **seg_components,
            "edge_bce":  edge_bce.item(),
            "edge_dice": edge_dice.item(),
            "total":     total.item(),
        }

        return total, components
