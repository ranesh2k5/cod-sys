"""
Training + Validation loop for CODNet.
Run: python train.py --config configs/default.yaml
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from model import build_model
from data.dataset import build_dataloaders
from training.losses import CODLoss
from evaluation.metrics import MetricAccumulator


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (imgs, masks, edges) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cpu"):
            seg_logits, edge_logits = model(imgs)
            loss, components = criterion(seg_logits, edge_logits, masks, edges)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"BCE: {components['bce']:.3f} | "
                  f"Dice: {components['dice']:.3f} | "
                  f"Time: {elapsed:.1f}s")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metrics = MetricAccumulator()
    total_loss = 0.0

    for imgs, masks, edges in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        seg_logits, edge_logits = model(imgs)
        loss, _ = criterion(seg_logits, edge_logits, masks, edges)
        total_loss += loss.item()

        probs = seg_logits.sigmoid()
        metrics.update(probs, masks)

    return total_loss / len(loader), metrics.compute()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Data ----------
    train_loader, val_loader = build_dataloaders(
        train_img_dir  = args.train_img,
        train_mask_dir = args.train_mask,
        val_img_dir    = args.val_img,
        val_mask_dir   = args.val_mask,
        img_size       = args.img_size,
        batch_size     = args.batch_size,
        num_workers    = args.workers,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ---------- Model ----------
    model = build_model(pretrained=True).to(device)

    # ---------- Loss ----------
    criterion = CODLoss(seg_weight=1.0, edge_weight=0.4)

    # ---------- Optimizer ----------
    # Lower LR for backbone (pretrained), higher for decoder (random init)
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.fpn.parameters()) + list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = GradScaler("cpu")

    # ---------- Training ----------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        iou = val_metrics["iou"]
        mae = val_metrics["mae"]
        fm  = val_metrics["f_measure"]

        print(f"\nEpoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"IoU: {iou:.4f} | F-measure: {fm:.4f} | MAE: {mae:.4f}\n")

        # Save best checkpoint
        if iou > best_iou:
            best_iou = iou
            ckpt_path = save_dir / "best.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "iou":         iou,
            }, ckpt_path)
            print(f"  ✓ Saved best model → {ckpt_path}  (IoU={iou:.4f})")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, save_dir / f"epoch_{epoch:03d}.pth")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CODNet Training")

    # Paths
    parser.add_argument("--train_img",  required=True, help="Path to train images dir")
    parser.add_argument("--train_mask", required=True, help="Path to train masks dir")
    parser.add_argument("--val_img",    required=True, help="Path to val images dir")
    parser.add_argument("--val_mask",   required=True, help="Path to val masks dir")
    parser.add_argument("--save_dir",   default="checkpoints", help="Where to save weights")

    # Hyperparams
    parser.add_argument("--img_size",   type=int,   default=384)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)

    args = parser.parse_args()
    main(args)
