"""Generate confusion matrix plots and per-class performance summaries."""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO


def find_best_model() -> str:
    """Return the newest trained model checkpoint."""
    patterns = [
        "runs/train/production_yolov8s*/weights/best.pt",
        "runs/train/baseline_yolov8n*/weights/best.pt",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    raise FileNotFoundError("No trained model found. Train a baseline or production model first.")


def find_data_yaml() -> str:
    """Locate the dataset configuration file."""
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


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], save_path: Path) -> None:
    """Render and persist the confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    matrix_norm = matrix.astype(float) / (matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
    sns.heatmap(
        matrix_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names + ["Background"],
        yticklabels=class_names + ["Background"],
        cbar_kws={"label": "Normalised Count"},
    )
    plt.title("Confusion Matrix (row-normalised)", fontsize=14, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def summarise_per_class(metrics, class_names: list[str]) -> list[dict[str, float]]:
    """Extract per-class AP@0.5 values where available."""
    if not hasattr(metrics, "box"):
        return []

    ap_class_index = getattr(metrics.box, "ap_class_index", [])
    ap50_values = getattr(metrics.box, "ap50", [])
    summary = []
    for idx, class_idx in enumerate(ap_class_index):
        if class_idx < len(class_names):
            ap50 = float(ap50_values[idx]) if idx < len(ap50_values) else 0.0
            summary.append({"class": class_names[class_idx], "ap50": ap50})
    return summary


def write_report(report_path: Path, metrics, class_names: list[str], class_summary: list[dict[str, float]]) -> None:
    """Persist a plain-text performance report."""
    box_metrics = getattr(metrics, "box", None)
    map50 = getattr(box_metrics, "map50", 0.0) if box_metrics else 0.0
    precision = getattr(box_metrics, "mp", 0.0) if box_metrics else 0.0
    recall = getattr(box_metrics, "mr", 0.0) if box_metrics else 0.0

    report_lines = []
    report_lines.append("DETAILED PERFORMANCE REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Overall mAP@0.5: {map50:.4f}")
    report_lines.append(f"Overall Precision: {precision:.4f}")
    report_lines.append(f"Overall Recall: {recall:.4f}")
    report_lines.append("")
    report_lines.append("Per-class mAP@0.5")
    report_lines.append("-" * 60)
    if class_summary:
        for item in class_summary:
            report_lines.append(f"{item['class']:<30} {item['ap50']:.4f}")
    else:
        report_lines.append("Per-class metrics not provided by the installed Ultralytics version.")

    weak = [item for item in class_summary if item["ap50"] < 0.30]
    moderate = [item for item in class_summary if 0.30 <= item["ap50"] < 0.60]

    report_lines.append("")
    report_lines.append("Recommendations")
    report_lines.append("-" * 60)
    if weak:
        report_lines.append("Classes below 30% AP@0.5:")
        for item in weak:
            report_lines.append(f"  - {item['class']}")
        report_lines.append("  Suggested actions: gather more samples, strengthen augmentation, review labels.")

    if moderate:
        report_lines.append("")
        report_lines.append("Classes between 30% and 60% AP@0.5:")
        for item in moderate:
            report_lines.append(f"  - {item['class']}")

    report_path.write_text("\n".join(report_lines))
    print(f"Saved detailed report to {report_path}")


def run_confusion(model_path: Optional[str], split: str) -> int:
    """Core execution routine. Returns 0 on success, non-zero on failure."""
    try:
        selected_model = Path(model_path) if model_path else Path(find_best_model())
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    if not selected_model.exists():
        print(f"ERROR: model not found at {selected_model}")
        return 1

    try:
        data_yaml = find_data_yaml()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Using model: {selected_model}")
    print(f"Dataset config: {data_yaml}")

    model = YOLO(str(selected_model))
    metrics = model.val(data=data_yaml, split=split, plots=False)

    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is None or confusion.matrix is None:
        print("Ultralytics did not return a confusion matrix for this run.")
        return 1

    matrix = confusion.matrix
    class_names = list(getattr(metrics, "names", {}).values())

    output_dir = selected_model.parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(matrix, class_names, cm_path)

    class_summary = summarise_per_class(metrics, class_names)
    if class_summary:
        print("\nPer-class AP@0.5")
        print("=" * 60)
        for item in class_summary:
            print(f"{item['class']:<25} {item['ap50']:.1%}")
        weak = [item for item in class_summary if item["ap50"] < 0.30]
        moderate = [item for item in class_summary if 0.30 <= item["ap50"] < 0.60]
        strong = [item for item in class_summary if item["ap50"] >= 0.60]

        if weak:
            print("\nClasses below 30% AP@0.5:")
            for item in weak:
                print(f"  - {item['class']} ({item['ap50']:.1%})")
        if moderate:
            print("\nClasses between 30% and 60% AP@0.5:")
            for item in moderate:
                print(f"  - {item['class']} ({item['ap50']:.1%})")
        if strong and not weak and not moderate:
            print("\nAll classes meet or exceed 60% AP@0.5.")
    else:
        print("Per-class AP@0.5 metrics are unavailable in this Ultralytics build.")

    report_path = output_dir / "performance_report.txt"
    write_report(report_path, metrics, class_names, class_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_cm = log_dir / f"confusion_matrix_{timestamp}.png"
    log_report = log_dir / f"performance_analysis_{timestamp}.txt"
    Path(log_cm).write_bytes(cm_path.read_bytes())
    Path(log_report).write_text(report_path.read_text())

    print("\nCopied artefacts to logs directory:")
    print(f"  {log_cm}")
    print(f"  {log_report}")

    return 0


def main(args: Optional[Iterable[str]] = None) -> int:
    """CLI entry point for confusion-matrix analysis."""
    parser = argparse.ArgumentParser(description="Generate confusion matrix and per-class metrics")
    parser.add_argument("model", nargs="?", help="Path to trained weights (defaults to newest run)")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parsed = parser.parse_args(args=args)
    return run_confusion(parsed.model, parsed.split)


if __name__ == "__main__":
    raise SystemExit(main())