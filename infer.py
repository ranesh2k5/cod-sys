"""
Inference script: takes an image → outputs segmentation mask + overlay.

Usage:
  python infer.py --checkpoint checkpoints/best.pth --input image.jpg
  python infer.py --checkpoint checkpoints/best.pth --input folder/  --output results/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import build_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} | IoU: {ckpt.get('iou', '?'):.4f}")
    return model


def preprocess(img_bgr: np.ndarray, img_size: int = 384):
    """BGR numpy → normalised tensor [1, 3, H, W]"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = TRANSFORM(img_resized)             # [3, H, W]
    return tensor.unsqueeze(0), img_rgb         # [1, 3, H, W], original RGB


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: torch.device,
            orig_size: tuple, threshold: float = 0.5) -> np.ndarray:
    """
    Returns float mask [0,1] resized to original image dimensions.
    """
    tensor = tensor.to(device)
    seg_logits, _ = model(tensor)
    prob = seg_logits.sigmoid()

    # Resize back to original
    prob = F.interpolate(prob, size=orig_size, mode="bilinear", align_corners=False)
    mask = (prob.squeeze().cpu().numpy())       # H x W float [0,1]
    return mask


def create_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                   threshold: float = 0.5, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays coloured mask on the original image.
      - Green tint for detected object
      - Red contour for boundary
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    binary  = (mask > threshold).astype(np.uint8) * 255

    # Green fill overlay
    overlay = img_bgr.copy()
    overlay[binary == 255] = (
        overlay[binary == 255] * (1 - alpha) + np.array([0, 200, 0]) * alpha
    ).astype(np.uint8)

    # Red contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    return overlay


def save_results(output_dir: Path, stem: str,
                 overlay: np.ndarray, mask: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
    mask_u8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask_u8)
    print(f"  Saved: {stem}_overlay.png  |  {stem}_mask.png")


def run_single(model, img_path: str, output_dir: Path,
               device: torch.device, img_size: int, threshold: float):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Cannot read: {img_path}")
        return

    h, w = img_bgr.shape[:2]
    tensor, img_rgb = preprocess(img_bgr, img_size)
    mask = predict(model, tensor, device, (h, w), threshold)
    overlay = create_overlay(img_rgb, mask, threshold)

    stem = Path(img_path).stem
    save_results(output_dir, stem, overlay, mask)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    output_dir = Path(args.output)
    input_path = Path(args.input)

    if input_path.is_dir():
        img_paths = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.jpeg"))
        )
        print(f"Processing {len(img_paths)} images...")
        for p in img_paths:
            run_single(model, str(p), output_dir, device, args.img_size, args.threshold)
    else:
        run_single(model, str(input_path), output_dir, device, args.img_size, args.threshold)

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CODNet Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--input",      required=True, help="Image path or folder")
    parser.add_argument("--output",     default="results", help="Output folder")
    parser.add_argument("--img_size",   type=int,   default=384)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()
    main(args)
