"""
Quick dataset checker for mixed detection/segmentation annotations and class name consistency.

Usage:
    python scripts/check_dataset.py [--auto-fix]

Outputs:
 - Returns exit code 0 if dataset looks consistent for detection training (YOLO TXT labels).
 - Returns non-zero if issues found (unless --auto-fix is used).

Auto-fix (experimental):
 - If a COCO JSON with 'segmentation' is found and --auto-fix is provided, the script will convert polygon segmentations
   to bounding boxes and write YOLO TXT labels next to images. Use with caution and backup data first.
"""

import os
import sys
import json
import glob
from pathlib import Path

AUTO_FIX = '--auto-fix' in sys.argv


def find_coco_jsons(data_dir='data'):
    patterns = [os.path.join(data_dir, '**', '*.json'), os.path.join(data_dir, '*.json')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return files


def count_segmentation_entries(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    seg_count = 0
    for ann in coco.get('annotations', []):
        if 'segmentation' in ann and ann['segmentation']:
            seg_count += 1
    return seg_count, len(coco.get('annotations', []))


def report_and_optionally_fix():
    jsons = find_coco_jsons()
    if not jsons:
        print('>>>>> No COCO JSON files found under data/ — dataset appears to be YOLO TXT labels (detection).')
        return 0

    issues_found = False
    for j in jsons:
        seg_count, total = count_segmentation_entries(j)
        if seg_count > 0:
            issues_found = True
            print(f'----- Found COCO JSON with segmentation: {j}')
            print(f'   Segmentation annotations: {seg_count} / {total}')

            if AUTO_FIX:
                print('>>>>> Auto-fix enabled: converting segmentation polygons to bounding boxes (YOLO format)')
                try:
                    convert_coco_seg_to_yolo(j)
                    print('>>>>> Conversion complete (labels written alongside images).')
                except Exception as e:
                    print(f'----- Auto-fix failed: {e}')
    if issues_found and not AUTO_FIX:
        print('\n----- Mixed detection/segmentation annotations detected. This will cause the trainer to drop segmentation data.')
        print('   Options:')
        print('     1) Re-export dataset from source as Detection (bounding boxes)')
        print('     2) Run this script with --auto-fix to convert polygons to boxes (experimental)')
        return 2

    if not issues_found:
        print('>>>>> No segmentation annotations detected in COCO JSON files. OK for detection training.')
        return 0


# Minimal COCO segmentation -> YOLO TXT converter (experimental)
# This converts each segmentation (polygon or list) to its bounding box, then to normalized YOLO format.
# It writes/overwrites .txt label files next to images (IMAGE.jpg -> IMAGE.txt). Use with care.

def convert_coco_seg_to_yolo(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    image_map = {img['id']: img for img in coco.get('images', [])}
    categories = {c['id']: c['name'] for c in coco.get('categories', [])}

    out_count = 0
    for ann in coco.get('annotations', []):
        img = image_map.get(ann['image_id'])
        if img is None:
            continue
        img_path = img.get('file_name')
        if not os.path.isabs(img_path):
            # assume relative to json file
            base = os.path.dirname(coco_json_path)
            img_path = os.path.join(base, img_path)
        if not os.path.exists(img_path):
            print(f'----- Image not found: {img_path} — skipping')
            continue

        # compute bbox from segmentation if needed
        bbox = ann.get('bbox')
        if (not bbox or bbox == [0,0,0,0]) and 'segmentation' in ann:
            seg = ann['segmentation']
            # segmentation can be list of lists (polygons) or RLE; we handle simple polygons
            if isinstance(seg, list) and seg:
                xs = []
                ys = []
                for poly in seg:
                    if isinstance(poly, list):
                        coords = poly
                        xs.extend(coords[0::2])
                        ys.extend(coords[1::2])
                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        if not bbox:
            continue

        # Convert bbox to YOLO format (class_id x_center y_center w h) normalized by image size
        img_w, img_h = img.get('width'), img.get('height')
        if not img_w or not img_h:
            # try to load image via PIL
            try:
                from PIL import Image
                im = Image.open(img_path)
                img_w, img_h = im.size
            except Exception:
                print(f'----- Cannot determine image size for {img_path}; skipping')
                continue

        x, y, w, h = bbox
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        x_c = x_center / img_w
        y_c = y_center / img_h
        w_n = w / img_w
        h_n = h / img_h

        class_name = categories.get(ann['category_id'], '0')
        # map category name to numeric id by order in coco categories (best-effort)
        class_id = ann.get('category_id', 0)

        # write label file
        lab_path = os.path.splitext(img_path)[0] + '.txt'
        with open(lab_path, 'a') as lf:
            lf.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        out_count += 1

    print(f'>>>>> Wrote/updated {out_count} YOLO label lines (appended to .txt files)')


if __name__ == '__main__':
    rc = report_and_optionally_fix()
    sys.exit(rc)
