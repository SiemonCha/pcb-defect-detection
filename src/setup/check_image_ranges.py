"""
Image range checker and optional fixer.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


def find_images(data_dir: str = "data") -> List[str]:
    """Locate image files under the dataset directory."""
    patterns = [
        os.path.join(data_dir, "**", "*.jpg"),
        os.path.join(data_dir, "**", "*.png"),
    ]
    images: List[str] = []
    for pattern in patterns:
        images.extend(glob.glob(pattern, recursive=True))
    return images


def check_and_fix(fix: bool = False) -> int:
    """Check image ranges and optionally convert float images to uint8."""
    images = find_images()
    if not images:
        print(">>>>> No images found under data/.")
        return 1

    issues: List[Tuple[str, str]] = []
    for image_path in images:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                issues.append((image_path, "cannot_read"))
                continue

            min_value = float(np.min(image))
            max_value = float(np.max(image))
            dtype = image.dtype

            if np.issubdtype(dtype, np.floating):
                issues.append((image_path, f"float_range={min_value:.6f}:{max_value:.6f}"))
                if fix:
                    corrected = np.clip(image, 0.0, 1.0)
                    corrected = (corrected * 255.0).astype(np.uint8)
                    cv2.imwrite(image_path, corrected)
                    print(f">>>>> Fixed float image (scaled to uint8): {image_path}")
            else:
                if max_value > 255 or min_value < 0:
                    issues.append((image_path, f"uint_range={min_value}:{max_value}"))
        except Exception as error:
            issues.append((image_path, f"error:{error}"))

    if issues:
        print("----- Image range issues detected (sample):")
        for image_path, message in issues[:20]:
            print(f"  {image_path}: {message}")
        if not fix:
            print("\n----- To attempt automatic fixes for float images, run with --fix")
            return 2
    else:
        print(">>>>> All images appear to be uint8 with values in [0,255]")

    return 0


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser(description="Check image pixel ranges and optionally fix float images.")
    parser.add_argument("--fix", action="store_true", help="Convert float images in [0,1] to uint8.")
    parsed_args = parser.parse_args(args=args)
    return check_and_fix(fix=parsed_args.fix)


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)

