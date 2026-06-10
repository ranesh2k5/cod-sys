"""
COD Dashboard Backend
Run: python app.py
Visit: http://localhost:5000
"""

import io
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms
from PIL import Image

# Import your model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model

app = Flask(__name__, static_folder="dashboard")
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = Path(__file__).parent / "best.pth"
IMG_SIZE   = 384
DEVICE     = torch.device("mps" if False
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Load model once at startup ────────────────────────────────────────────
print(f"Loading model on {DEVICE}...")
model = build_model(pretrained=False).to(DEVICE)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model ready (epoch {ckpt.get('epoch','?')}, IoU={ckpt.get('iou',0):.4f})")


# ── Inference helpers ─────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image):
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w    = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    tensor  = TRANSFORM(resized).unsqueeze(0).to(DEVICE)
    return tensor, img_rgb, (h, w)


@torch.no_grad()
def run_inference(tensor, orig_size, threshold=0.5):
    t0 = time.time()
    seg_logits, edge_logits = model(tensor)
    ms = (time.time() - t0) * 1000

    prob = seg_logits.sigmoid()
    prob = F.interpolate(prob, size=orig_size, mode="bilinear", align_corners=False)
    prob_np = prob.squeeze().cpu().numpy()

    edge_prob = edge_logits.sigmoid()
    edge_prob = F.interpolate(edge_prob, size=orig_size, mode="bilinear", align_corners=False)
    edge_np = edge_prob.squeeze().cpu().numpy()

    binary = (prob_np > threshold).astype(np.uint8)
    coverage = float(binary.mean() * 100)

    return prob_np, edge_np, binary, coverage, ms


def create_overlay(img_rgb, binary, prob_np, alpha=0.45):
    """Green overlay + red contour on original image."""
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay   = img_bgr.copy()
    mask_bool = binary.astype(bool)

    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + np.array([0, 200, 0]) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 230), 2)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def create_heatmap(prob_np):
    """Confidence heatmap."""
    heat_u8  = (prob_np * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def ndarray_to_b64(arr: np.ndarray, quality: int = 90) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file      = request.files["image"]
    threshold = float(request.form.get("threshold", 0.5))

    try:
        pil_img = Image.open(file.stream)
        tensor, img_rgb, orig_size = preprocess(pil_img)
        prob_np, edge_np, binary, coverage, ms = run_inference(tensor, orig_size, threshold)

        overlay  = create_overlay(img_rgb, binary, prob_np)
        heatmap  = create_heatmap(prob_np)
        mask_rgb = np.stack([binary * 255] * 3, axis=-1)

        # Metrics
        iou_score = float(ckpt.get("iou", 0))

        return jsonify({
            "overlay":    ndarray_to_b64(overlay),
            "heatmap":    ndarray_to_b64(heatmap),
            "mask":       ndarray_to_b64(mask_rgb),
            "original":   ndarray_to_b64(img_rgb),
            "coverage":   round(coverage, 2),
            "inference_ms": round(ms, 1),
            "model_iou":  round(iou_score, 4),
            "threshold":  threshold,
            "device":     str(DEVICE),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    return jsonify({
        "model": "CODNet (EfficientNet-B4)",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT),
        "iou": round(float(ckpt.get("iou", 0)), 4),
        "epoch": ckpt.get("epoch", "?"),
    })


if __name__ == "__main__":
    print(f"\n  COD Dashboard → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
