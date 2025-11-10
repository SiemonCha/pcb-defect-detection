"""
Dataset consistency checks and optional COCO-to-YOLO conversion.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple


def find_coco_jsons(data_dir: str = "data") -> List[str]:
    """Search for COCO JSON files within the dataset directory."""
    patterns = [os.path.join(data_dir, "**", "*.json"), os.path.join(data_dir, "*.json")]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def count_segmentation_entries(coco_json_path: str) -> Tuple[int, int]:
    """Count segmentation annotations in a COCO JSON file."""
    with open(coco_json_path, "r") as file:
        coco = json.load(file)
    segmentation_count = sum(
        1 for ann in coco.get("annotations", []) if ann.get("segmentation")
    )
    return segmentation_count, len(coco.get("annotations", []))


def convert_coco_seg_to_yolo(coco_json_path: str):
    """Convert COCO segmentation polygons to YOLO bounding box format."""
    with open(coco_json_path, "r") as file:
        coco = json.load(file)

    image_map = {img["id"]: img for img in coco.get("images", [])}
    categories = {category["id"]: category["name"] for category in coco.get("categories", [])}

    updated_count = 0
    for annotation in coco.get("annotations", []):
        image_info = image_map.get(annotation["image_id"])
        if not image_info:
            continue

        image_path = image_info.get("file_name")
        if not os.path.isabs(image_path):
            base = os.path.dirname(coco_json_path)
            image_path = os.path.join(base, image_path)
        if not os.path.exists(image_path):
            print(f"----- Image not found: {image_path} - skipping")
            continue

        bbox = annotation.get("bbox")
        if (not bbox or bbox == [0, 0, 0, 0]) and annotation.get("segmentation"):
            seg = annotation["segmentation"]
            if isinstance(seg, list) and seg:
                xs, ys = [], []
                for poly in seg:
                    if isinstance(poly, list):
                        xs.extend(poly[0::2])
                        ys.extend(poly[1::2])
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        if not bbox:
            continue

        img_w, img_h = image_info.get("width"), image_info.get("height")
        if not img_w or not img_h:
            try:
                from PIL import Image

                width, height = Image.open(image_path).size
                img_w, img_h = width, height
            except Exception:
                print(f"----- Cannot determine image size for {image_path}; skipping")
                continue

        x, y, w, h = bbox
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        x_c = x_center / img_w
        y_c = y_center / img_h
        w_n = w / img_w
        h_n = h / img_h

        class_id = annotation.get("category_id", 0)
        _ = categories.get(class_id, "0")  # currently unused but kept for clarity

        label_path = os.path.splitext(image_path)[0] + ".txt"
        with open(label_path, "a") as label_file:
            label_file.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        updated_count += 1

    print(f">>>>> Wrote/updated {updated_count} YOLO label lines (appended to .txt files)")


def report_and_optionally_fix(auto_fix: bool = False) -> int:
    """Report dataset issues and optionally convert segmentation annotations."""
    json_files = find_coco_jsons()
    if not json_files:
        print(">>>>> No COCO JSON files found under data/ - dataset appears to use YOLO TXT labels.")
        return 0

    issues_found = False
    for json_file in json_files:
        seg_count, total = count_segmentation_entries(json_file)
        if seg_count > 0:
            issues_found = True
            print(f"----- Found COCO JSON with segmentation: {json_file}")
            print(f"   Segmentation annotations: {seg_count} / {total}")

            if auto_fix:
                print(">>>>> Auto-fix enabled: converting segmentation polygons to bounding boxes (YOLO format)")
                try:
                    convert_coco_seg_to_yolo(json_file)
                    print(">>>>> Conversion complete (labels written alongside images).")
                except Exception as error:
                    print(f"----- Auto-fix failed: {error}")

    if issues_found and not auto_fix:
        print("\n----- Mixed detection/segmentation annotations detected. This will cause segmentation data to be dropped.")
        print("   Options:")
        print("     1) Re-export dataset from source as Detection (bounding boxes)")
        print("     2) Run this script with --auto-fix to convert polygons to boxes (experimental)")
        return 2

    if not issues_found:
        print(">>>>> No segmentation annotations detected in COCO JSON files. OK for detection training.")
    return 0


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser(description="Check dataset annotations for YOLO detection training.")
    parser.add_argument("--auto-fix", action="store_true", help="Convert segmentation polygons to YOLO boxes.")
    parsed_args = parser.parse_args(args=args)
    return report_and_optionally_fix(auto_fix=parsed_args.auto_fix)


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)

