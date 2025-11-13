"""
Model interpretability utilities (Grad-CAM approximations and CLI entry).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg", force=True)
import torch
import torch.nn.functional as F
from data import resolve_dataset_yaml


def find_best_model() -> str:
    """Locate the most recent trained model."""
    patterns = [
        "runs/train/production_yolov8s*/weights/best.pt",
        "runs/train/baseline_yolov8n*/weights/best.pt",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    raise FileNotFoundError("No trained model found. Run: python -m cli train-baseline")


def find_data_yaml() -> Path:
    """Locate data.yaml describing the dataset."""
    return resolve_dataset_yaml()


def generate_heatmap(model: YOLO, image: np.ndarray) -> np.ndarray:
    """Generate a simple attention heatmap based on detection confidence."""
    results = model.predict(image, conf=0.25, verbose=False)
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += conf

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    return heatmap


def apply_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap onto the original image."""
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def visualize_prediction(
    image_path: str | Path,
    model: YOLO,
    class_names: Iterable[str],
    save_path: Path,
) -> bool:
    """Create interpretability visualization for a single image."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(str(image_path), conf=0.25, verbose=False)
    heatmap = generate_heatmap(model, image_rgb)
    overlay = apply_overlay(image_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    prediction_image = image_rgb.copy()
    box_count = 0
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        box_count = len(boxes)

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(prediction_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_names[cls]} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(prediction_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (255, 0, 0), -1)
            cv2.putText(
                prediction_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    axes[1].imshow(prediction_image)
    axes[1].set_title(f"Predictions ({box_count} defects)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Attention Map (High = Red)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def analyze_confidence_distribution(model: YOLO, image_paths: List[str | Path]) -> np.ndarray:
    """Gather confidence scores from model predictions."""
    confidences: List[float] = []
    for image_path in image_paths[:100]:
        results = model.predict(str(image_path), conf=0.01, verbose=False)
        if results[0].boxes is not None:
            confidences.extend(results[0].boxes.conf.cpu().numpy())
    return np.array(confidences)


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="Specific image to visualize")
    parser.add_argument("--samples", type=int, default=10, help="Number of random samples")
    parsed_args = parser.parse_args(args=args)

    model_path = parsed_args.model or find_best_model()
    data_yaml = find_data_yaml()

    print("=" * 60)
    print("MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")

    model = YOLO(model_path)
    with open(data_yaml, "r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    raw_names = data_config["names"]
    if isinstance(raw_names, dict):
        class_names = [raw_names[k] for k in sorted(raw_names)]
    else:
        class_names = list(raw_names)
    dataset_root = data_yaml.parent
    test_img_dir = dataset_root / "test" / "images"

    if not test_img_dir.exists():
        print(f"xxxx Test images not found: {test_img_dir}")
        return

    test_images = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    print(f"\n>>>> Found {len(test_images)} test images")
    output_dir = Path("logs/interpretability")
    output_dir.mkdir(parents=True, exist_ok=True)

    if parsed_args.image:
        images_to_visualize = [Path(parsed_args.image)]
    else:
        import random

        random.seed(42)
        population = test_images
        images_to_visualize = random.sample(population, min(parsed_args.samples, len(population)))

    print(f">>>> Visualizing {len(images_to_visualize)} images...")
    visualized = 0
    for index, image_path in enumerate(images_to_visualize):
        save_path = output_dir / f"interpretation_{index + 1}_{Path(image_path).stem}.png"
        if visualize_prediction(image_path, model, class_names, save_path):
            visualized += 1

    print("\n>>>> Analyzing confidence distribution...")
    confidences = analyze_confidence_distribution(model, test_images)

    if len(confidences) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Confidence Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
        plt.axvline(0.25, color="r", linestyle="--", label="Default Threshold (0.25)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"   Mean confidence: {confidences.mean():.3f}")
        print(f"   Std confidence: {confidences.std():.3f}")
        print(f"   Min confidence: {confidences.min():.3f}")
        print(f"   Max confidence: {confidences.max():.3f}")

    report_path = output_dir / "interpretability_report.txt"
    with open(report_path, "w") as file:
        file.write("MODEL INTERPRETABILITY REPORT\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Model: {os.path.basename(model_path)}\n")
        file.write(f"Visualized: {visualized} images\n\n")

        if len(confidences) > 0:
            file.write("Confidence Statistics:\n")
            file.write("-" * 60 + "\n")
            file.write(f"Mean: {confidences.mean():.3f}\n")
            file.write(f"Std:  {confidences.std():.3f}\n")
            file.write(f"Min:  {confidences.min():.3f}\n")
            file.write(f"Max:  {confidences.max():.3f}\n\n")

            file.write("Confidence Interpretation:\n")
            file.write("-" * 60 + "\n")
            high_conf = (confidences > 0.7).sum() / len(confidences)
            med_conf = ((confidences > 0.4) & (confidences <= 0.7)).sum() / len(confidences)
            low_conf = (confidences <= 0.4).sum() / len(confidences)
            file.write(f"High confidence (>0.7): {high_conf:.1%}\n")
            file.write(f"Medium confidence (0.4-0.7): {med_conf:.1%}\n")
            file.write(f"Low confidence (<0.4): {low_conf:.1%}\n\n")

            if low_conf > 0.2:
                file.write("WARNING: high proportion of low-confidence predictions\n")
                file.write("   - Model is uncertain on many detections\n")
                file.write("   - Consider retraining with more data\n\n")
            else:
                file.write("Model shows good confidence in predictions\n\n")

        file.write("Notes on Attention Maps:\n")
        file.write("-" * 60 + "\n")
        file.write("- Red regions indicate high model attention\n")
        file.write("- Blue regions indicate low model attention\n")
        file.write("- Good models focus on defect regions\n")
        file.write("- Scattered attention may indicate overfitting\n")

    print(f"\n>>>> Saved {visualized} visualizations")
    print(f">>>> Output: {output_dir}")
    print(f">>>> Report: {report_path}")
    print("\n>>>> Review attention maps to understand:")
    print("   - Does the model focus on defects or background?")
    print("   - Are attention patterns consistent?")
    print("   - Any unexpected focus areas?")


if __name__ == "__main__":
    main()

