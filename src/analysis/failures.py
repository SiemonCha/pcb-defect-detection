"""
Failure case analysis and visualization.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


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


def find_data_yaml() -> str:
    """Locate data.yaml describing the dataset."""
    dataset_path_file = Path("dataset_path.txt")
    if dataset_path_file.exists():
        dataset_path = dataset_path_file.read_text().strip()
        data_yaml = Path(dataset_path) / "data.yaml"
        if data_yaml.exists():
            return str(data_yaml)

    patterns = ["data/*/data.yaml", "data/data.yaml"]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError("data.yaml not found. Run: python -m cli data-download")


def parse_yolo_label(label_path: str, img_shape: Tuple[int, int, int]) -> List[Dict]:
    """Parse YOLO format labels to bounding boxes."""
    height, width = img_shape[:2]
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:5])

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            boxes.append(
                {
                    "class": cls_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 1.0,  # Ground truth
                }
            )

    return boxes


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def analyze_prediction(
    pred_boxes: List[Dict], gt_boxes: List[Dict], iou_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match predictions to ground truth and categorize failures."""
    matched_gt = set()
    matched_pred = set()
    true_positives = []

    for i, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            if j in matched_gt or pred["class"] != gt["class"]:
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            matched_pred.add(i)
            matched_gt.add(best_gt_idx)
            true_positives.append({"pred": pred, "gt": gt_boxes[best_gt_idx], "iou": best_iou})

    false_positives = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    false_negatives = [gt_boxes[j] for j in range(len(gt_boxes)) if j not in matched_gt]

    return false_positives, false_negatives, true_positives


def visualize_failure(
    img_path: str,
    pred_boxes: List[Dict],
    gt_boxes: List[Dict],
    failure_type: str,
    class_names: List[str],
    save_path: Path,
) -> bool:
    """Create side-by-side visualization for a failure case."""
    image = cv2.imread(img_path)
    if image is None:
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(image)
    ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
    ax1.axis("off")

    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["bbox"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green", linewidth=2)
        ax1.add_patch(rect)
        ax1.text(
            x1,
            y1 - 5,
            class_names[gt["class"]],
            color="white",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="green", alpha=0.8),
        )

    ax2.imshow(image)
    ax2.set_title(f"Prediction - {failure_type}", fontsize=14, fontweight="bold")
    ax2.axis("off")

    for pred in pred_boxes:
        x1, y1, x2, y2 = pred["bbox"]
        color = "red" if failure_type == "False Positive" else "blue"
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2, linestyle="--"
        )
        ax2.add_patch(rect)
        label = f"{class_names[pred['class']]} {pred['confidence']:.2f}"
        ax2.text(
            x1,
            y1 - 5,
            label,
            color="white",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--top", type=int, default=20, help="Number of failures to visualize")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parsed_args = parser.parse_args(args=args)

    model_path = parsed_args.model or find_best_model()
    data_yaml = find_data_yaml()

    print("=" * 60)
    print("FAILURE CASE ANALYSIS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")

    model = YOLO(model_path)
    with open(data_yaml, "r") as file:
        data_config = yaml.safe_load(file)

    class_names = data_config["names"]
    dataset_root = os.path.dirname(data_yaml)
    test_img_dir = os.path.join(dataset_root, "test", "images")
    test_label_dir = os.path.join(dataset_root, "test", "labels")

    if not os.path.exists(test_img_dir):
        print(f"xxxx Test images not found: {test_img_dir}")
        return

    test_images = glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(
        os.path.join(test_img_dir, "*.png")
    )

    print(f"\n>>>> Found {len(test_images)} test images")
    print(">>>> Analyzing failures...")

    failures: Dict[str, List[Dict]] = {
        "false_positives": [],
        "false_negatives": [],
        "low_iou": [],
    }

    for img_path in test_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(test_label_dir, f"{img_name}.txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        gt_boxes = parse_yolo_label(label_path, image.shape)

        results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
        pred_boxes: List[Dict] = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, classes):
                pred_boxes.append(
                    {
                        "class": cls,
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                    }
                )

        fps, fns, tps = analyze_prediction(pred_boxes, gt_boxes, parsed_args.iou_threshold)

        if fps:
            failures["false_positives"].append(
                {
                    "img_path": img_path,
                    "pred_boxes": fps,
                    "gt_boxes": gt_boxes,
                    "count": len(fps),
                }
            )

        if fns:
            failures["false_negatives"].append(
                {
                    "img_path": img_path,
                    "pred_boxes": pred_boxes,
                    "gt_boxes": fns,
                    "count": len(fns),
                }
            )

        for tp in tps:
            if tp["iou"] < 0.7:
                failures["low_iou"].append(
                    {
                        "img_path": img_path,
                        "pred_boxes": [tp["pred"]],
                        "gt_boxes": [tp["gt"]],
                        "iou": tp["iou"],
                    }
                )

    print(f"\n{'='*60}")
    print("FAILURE STATISTICS")
    print(f"{'='*60}")
    print(f"False Positives: {len(failures['false_positives'])} images")
    print(f"False Negatives: {len(failures['false_negatives'])} images")
    print(f"Low IoU (<0.7): {len(failures['low_iou'])} detections")

    output_dir = Path("logs/failure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>>> Visualizing top {parsed_args.top} failures...")
    visualized = 0

    fp_sorted = sorted(failures["false_positives"], key=lambda item: item["count"], reverse=True)
    for idx, failure in enumerate(fp_sorted[: parsed_args.top // 2]):
        save_path = output_dir / f"false_positive_{idx + 1}.png"
        if visualize_failure(
            failure["img_path"],
            failure["pred_boxes"],
            failure["gt_boxes"],
            "False Positive",
            class_names,
            save_path,
        ):
            visualized += 1

    fn_sorted = sorted(failures["false_negatives"], key=lambda item: item["count"], reverse=True)
    for idx, failure in enumerate(fn_sorted[: parsed_args.top // 2]):
        save_path = output_dir / f"false_negative_{idx + 1}.png"
        if visualize_failure(
            failure["img_path"],
            failure["pred_boxes"],
            failure["gt_boxes"],
            "False Negative (Missed)",
            class_names,
            save_path,
        ):
            visualized += 1

    report_path = output_dir / "failure_report.txt"
    with open(report_path, "w") as file:
        file.write("FAILURE CASE ANALYSIS REPORT\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Model: {os.path.basename(model_path)}\n")
        file.write(f"Test Images: {len(test_images)}\n")
        file.write(f"IoU Threshold: {parsed_args.iou_threshold}\n\n")

        file.write("Failure Statistics:\n")
        file.write("-" * 60 + "\n")
        file.write(f"Images with False Positives: {len(failures['false_positives'])}\n")
        file.write(f"Images with False Negatives: {len(failures['false_negatives'])}\n")
        file.write(f"Low IoU Detections (<0.7): {len(failures['low_iou'])}\n\n")

        file.write("Recommendations:\n")
        file.write("-" * 60 + "\n")
        if len(failures["false_positives"]) > len(test_images) * 0.1:
            file.write("WARNING: high false positive rate (>10%)\n")
            file.write("   - Increase confidence threshold\n")
            file.write("   - Add hard negative mining\n\n")

        if len(failures["false_negatives"]) > len(test_images) * 0.1:
            file.write("WARNING: high false negative rate (>10%)\n")
            file.write("   - Lower confidence threshold\n")
            file.write("   - Collect more training samples for missed classes\n\n")

        if len(failures["low_iou"]) > 50:
            file.write("WARNING: many low IoU detections\n")
            file.write("   - Model localizes poorly\n")
            file.write("   - Increase training epochs or use a larger model\n\n")

        if not failures["false_positives"] and not failures["false_negatives"]:
            file.write("Performance is strong with few failures.\n")

    print(f"\n>>>> Visualized {visualized} failure cases")
    print(f">>>> Saved to: {output_dir}")
    print(f">>>> Report: {report_path}")


if __name__ == "__main__":
    main()

