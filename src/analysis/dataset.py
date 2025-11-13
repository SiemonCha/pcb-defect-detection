"""
Dataset analysis utilities and CLI entry point.
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data import resolve_dataset_yaml


def find_data_yaml() -> Path:
    """Locate data.yaml describing the dataset using config + fallbacks."""
    return resolve_dataset_yaml()


def analyze_split(label_dir: str, split_name: str) -> Optional[Dict]:
    """Analyze a dataset split."""
    if not os.path.exists(label_dir):
        return None

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    class_counts = Counter()
    bbox_counts = []
    bbox_areas = []

    for label_file in label_files:
        with open(label_file, "r") as file:
            lines = file.readlines()
            bbox_counts.append(len(lines))

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

                    width = float(parts[3])
                    height = float(parts[4])
                    bbox_areas.append(width * height)

    return {
        "num_images": len(label_files),
        "class_counts": class_counts,
        "bbox_per_image": bbox_counts,
        "bbox_areas": bbox_areas,
    }


def calculate_imbalance_metrics(class_counts: Counter, num_classes: int) -> Dict:
    """Calculate class imbalance metrics."""
    counts = [class_counts.get(i, 0) for i in range(num_classes)]
    if not any(counts):
        return {}

    max_count = max(counts)
    non_zero_counts = [c for c in counts if c > 0]
    min_count = min(non_zero_counts) if non_zero_counts else 0

    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    gini = (
        (2 * sum((i + 1) * counts_sorted[i] for i in range(n)))
        / (n * sum(counts_sorted))
        - (n + 1) / n
        if sum(counts_sorted) > 0
        else 0
    )

    return {
        "imbalance_ratio": imbalance_ratio,
        "gini_coefficient": gini,
        "max_count": max_count,
        "min_count": min_count,
        "counts": counts,
    }


def main():
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    data_yaml = find_data_yaml()
    dataset_root = data_yaml.parent

    with open(data_yaml, "r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    class_names_raw = data_config["names"]
    if isinstance(class_names_raw, dict):
        class_names = [class_names_raw[k] for k in sorted(class_names_raw)]
    else:
        class_names = list(class_names_raw)
    num_classes = data_config.get("nc", len(class_names))

    print(f"\nDataset: {data_yaml}")
    print(f"Classes ({num_classes}): {class_names}")

    splits = ["train", "valid", "test"]
    split_stats: Dict[str, Dict] = {}

    for split in splits:
        label_dir = os.path.join(dataset_root, split, "labels")
        stats = analyze_split(label_dir, split)
        if not stats:
            continue

        split_stats[split] = stats
        print(f"\n{'='*60}")
        print(f"{split.upper()} SPLIT")
        print(f"{'='*60}")
        print(f"Images: {stats['num_images']}")
        print(f"Total annotations: {sum(stats['class_counts'].values())}")
        print(f"Avg bboxes per image: {np.mean(stats['bbox_per_image']):.2f}")

        print("\nClass distribution:")
        for class_id in range(num_classes):
            count = stats["class_counts"].get(class_id, 0)
            total = sum(stats["class_counts"].values()) or 1
            percentage = count / total * 100
            print(f"  {class_names[class_id]:20s}: {count:5d} ({percentage:5.1f}%)")

    print(f"\n{'='*60}")
    print("CLASS IMBALANCE ANALYSIS")
    print(f"{'='*60}")

    overall_counts = Counter()
    for stats in split_stats.values():
        overall_counts.update(stats["class_counts"])

    imbalance = calculate_imbalance_metrics(overall_counts, num_classes)

    if imbalance:
        max_index = imbalance["counts"].index(imbalance["max_count"])
        min_index = imbalance["counts"].index(imbalance["min_count"])
        print(f"\nImbalance Ratio (max/min): {imbalance['imbalance_ratio']:.2f}")
        print(f"Gini Coefficient: {imbalance['gini_coefficient']:.3f}")
        print(f"Most common class: {class_names[max_index]} ({imbalance['max_count']} samples)")
        print(f"Least common class: {class_names[min_index]} ({imbalance['min_count']} samples)")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    if imbalance and imbalance["imbalance_ratio"] > 10:
        print("\nSevere class imbalance (ratio > 10)")
        print("   Recommendation: collect more data for minority classes")
    elif imbalance and imbalance["imbalance_ratio"] > 5:
        print("\nModerate class imbalance (ratio > 5)")
        print("   Recommendation: apply class weights or augmentation")
    else:
        print("\nBalanced dataset (ratio < 5)")

    missing_classes = [i for i in range(num_classes) if overall_counts.get(i, 0) == 0]
    if missing_classes:
        missing_names = [class_names[i] for i in missing_classes]
        print(f"\nWarning: missing classes {missing_names}")
        print("   These classes have no training samples!")

    output_dir = Path("logs/dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, split in enumerate(["train", "valid", "test"]):
        if split not in split_stats:
            continue
        ax = axes[idx // 2, idx % 2]
        stats = split_stats[split]
        counts = [stats["class_counts"].get(i, 0) for i in range(num_classes)]
        x = np.arange(num_classes)
        bars = ax.bar(x, counts, color="steelblue", edgecolor="black")

        if counts:
            max_count = max(counts)
            for bar, count in zip(bars, counts):
                if count < max_count * 0.5:
                    bar.set_color("coral")

        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{split.title()} Split Distribution", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(num_classes)], rotation=45)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    counts = [overall_counts.get(i, 0) for i in range(num_classes)]
    x = np.arange(num_classes)
    bars = ax.bar(x, counts, color="steelblue", edgecolor="black")

    max_count = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        if count < max_count * 0.5:
            bar.set_color("coral")

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Overall Distribution", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_classes)], rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / "class_distribution.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    all_areas = []
    for stats in split_stats.values():
        all_areas.extend(stats["bbox_areas"])

    if all_areas:
        ax.hist(all_areas, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(all_areas), color="r", linestyle="--", label=f"Mean: {np.mean(all_areas):.3f}")
        ax.set_xlabel("Relative BBox Area (w x h)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Bounding Box Size Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        bbox_viz_path = output_dir / "bbox_size_distribution.png"
        plt.savefig(bbox_viz_path, dpi=150, bbox_inches="tight")
        plt.close()

    report_path = output_dir / f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as file:
        file.write("DATASET ANALYSIS REPORT\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Dataset: {data_yaml}\n")
        file.write(f"Classes: {num_classes}\n\n")

        for split in splits:
            if split not in split_stats:
                continue
            stats = split_stats[split]
            file.write(f"\n{split.upper()} Split:\n")
            file.write("-" * 60 + "\n")
            file.write(f"Images: {stats['num_images']}\n")
            file.write(f"Total annotations: {sum(stats['class_counts'].values())}\n")
            file.write(f"Avg bboxes/image: {np.mean(stats['bbox_per_image']):.2f}\n\n")

            file.write("Class distribution:\n")
            for class_id in range(num_classes):
                count = stats["class_counts"].get(class_id, 0)
                total = sum(stats["class_counts"].values()) or 1
                pct = count / total * 100
                file.write(f"  {class_names[class_id]:20s}: {count:5d} ({pct:5.1f}%)\n")

        if imbalance:
            file.write("\n\nClass Imbalance Metrics:\n")
            file.write("-" * 60 + "\n")
            file.write(f"Imbalance Ratio: {imbalance['imbalance_ratio']:.2f}\n")
            file.write(f"Gini Coefficient: {imbalance['gini_coefficient']:.3f}\n\n")

            if imbalance["imbalance_ratio"] > 10:
                file.write("Severe imbalance detected\n")
            elif imbalance["imbalance_ratio"] > 5:
                file.write("Moderate imbalance\n")
            else:
                file.write("Balanced dataset\n")

    print("\n>>>> Analysis complete!")
    print(f">>>> Visualizations: {output_dir}")
    print(f">>>> Report: {report_path}")


if __name__ == "__main__":
    main()

