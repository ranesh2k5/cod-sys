"""
Evaluation Metrics for Camouflaged Object Detection

Standard COD benchmark metrics:
  - Mean Absolute Error (MAE)    — average pixel-wise error
  - F-measure (weighted Fβ)      — precision-recall trade-off (β²=0.3, standard)
  - Intersection over Union (IoU) — overlap quality
  - S-measure (optional)         — structural similarity

All metrics operate on probability maps (0–1) OR binary masks.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-batch metric computation (PyTorch, differentiable-friendly)
# ---------------------------------------------------------------------------

def compute_iou(pred_prob: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Args:
        pred_prob: [B, 1, H, W] probability map
        target:    [B, 1, H, W] binary mask {0, 1}
    Returns:
        mean IoU over batch
    """
    pred_bin = (pred_prob > threshold).float()
    B = pred_bin.shape[0]

    pred_flat   = pred_bin.view(B, -1)
    target_flat = target.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou          = (intersection + smooth) / (union + smooth)
    return iou.mean()


def compute_mae(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error — pixel-wise average absolute difference."""
    return torch.abs(pred_prob - target.float()).mean()


def compute_fmeasure(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    beta_sq: float = 0.3,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted F-measure with β²=0.3 (COD standard: precision > recall).

    Fβ = (1 + β²) * precision * recall / (β² * precision + recall)
    """
    pred_bin = (pred_prob > threshold).float()
    B = pred_bin.shape[0]

    pred_flat   = pred_bin.view(B, -1)
    target_flat = target.float().view(B, -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = pred_flat.sum(dim=1) - tp
    fn = target_flat.sum(dim=1) - tp

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    fmeasure = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall + smooth)
    return fmeasure.mean()


# ---------------------------------------------------------------------------
# Running metric accumulator (for epoch-level reporting)
# ---------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """Accumulates metrics across batches, returns epoch averages."""
    iou_sum:   float = 0.0
    mae_sum:   float = 0.0
    f_sum:     float = 0.0
    n_batches: int   = 0

    def update(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        with torch.no_grad():
            self.iou_sum += compute_iou(pred_prob, target, threshold).item()
            self.mae_sum += compute_mae(pred_prob, target).item()
            self.f_sum   += compute_fmeasure(pred_prob, target, threshold=threshold).item()
            self.n_batches += 1

    def compute(self) -> dict:
        if self.n_batches == 0:
            return {"iou": 0.0, "mae": 0.0, "f_measure": 0.0}
        n = self.n_batches
        return {
            "iou":       self.iou_sum   / n,
            "mae":       self.mae_sum   / n,
            "f_measure": self.f_sum     / n,
        }

    def reset(self) -> None:
        self.iou_sum   = 0.0
        self.mae_sum   = 0.0
        self.f_sum     = 0.0
        self.n_batches = 0


# ---------------------------------------------------------------------------
# Threshold-independent: mean F-measure over thresholds (adaptive Fm)
# Standard in COD benchmarks
# ---------------------------------------------------------------------------

def adaptive_fmeasure(pred_prob: torch.Tensor, target: torch.Tensor,
                      num_thresh: int = 256, beta_sq: float = 0.3) -> float:
    """
    Compute F-measure at each threshold from 0 to 1 and return the maximum.
    This is the 'maxFm' metric used in official COD benchmarks.
    """
    thresholds = torch.linspace(0, 1, num_thresh, device=pred_prob.device)
    best_f = 0.0
    for t in thresholds:
        f = compute_fmeasure(pred_prob, target, beta_sq=beta_sq, threshold=t.item())
        if f.item() > best_f:
            best_f = f.item()
    return best_f


# ---------------------------------------------------------------------------
# Convenience: evaluate single image pair (numpy arrays)
# Used in inference / demo scripts
# ---------------------------------------------------------------------------

def evaluate_single(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Args:
        pred_mask: H×W float array [0, 1]
        gt_mask:   H×W binary array {0, 1}
    """
    pred_t = torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0)
    gt_t   = torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0)

    return {
        "iou":       compute_iou(pred_t, gt_t, threshold).item(),
        "mae":       compute_mae(pred_t, gt_t).item(),
        "f_measure": compute_fmeasure(pred_t, gt_t, threshold=threshold).item(),
        "max_fm":    adaptive_fmeasure(pred_t, gt_t),
    }
