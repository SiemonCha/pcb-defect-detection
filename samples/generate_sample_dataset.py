"""Generate a tiny synthetic dataset that mimics the PCB detection layout.

The generated structure is compatible with Ultralytics YOLO:

samples/
  sample_dataset/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/

Usage
-----
python samples/generate_sample_dataset.py --output samples/sample_dataset

By default the script overwrites any existing dataset at the output location.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

CLASSES = ["missing_hole", "open_circuit", "short" ]
SPLITS = {
    "train": 12,
    "valid": 4,
    "test": 4,
}
IMAGE_SIZE = (640, 640)
RNG = random.Random(42)


def _make_box() -> tuple[float, float, float, float]:
    """Return a bounding box (x_center, y_center, width, height) in YOLO format."""
    x_center = RNG.uniform(0.2, 0.8)
    y_center = RNG.uniform(0.2, 0.8)
    width = RNG.uniform(0.1, 0.3)
    height = RNG.uniform(0.1, 0.3)
    return x_center, y_center, width, height


def _draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], outline: str) -> None:
    x_c, y_c, w, h = box
    img_w, img_h = IMAGE_SIZE
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=4)


def _create_example(idx: int, split: str, out_dir: Path) -> None:
    image = Image.new("RGB", IMAGE_SIZE, color="#1e1e1e")
    draw = ImageDraw.Draw(image)

    num_boxes = RNG.randint(1, 3)
    label_lines: list[str] = []
    for _ in range(num_boxes):
        class_id = RNG.randrange(len(CLASSES))
        box = _make_box()
        colour = ["#ffb347", "#ff6961", "#77dd77"][class_id]
        _draw_box(draw, box, outline=colour)
        label_lines.append(
            f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        )

    file_name = f"pcb_{split}_{idx:03d}"
    image_path = out_dir / "images" / f"{file_name}.jpg"
    label_path = out_dir / "labels" / f"{file_name}.txt"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(image_path, quality=90)
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def _write_data_yaml(root: Path) -> None:
    data_yaml = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")


def generate(output: Path, overwrite: bool = True) -> Path:
    if output.exists() and overwrite:
        for child in sorted(output.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
        for child in sorted(output.glob("**/*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
    output.mkdir(parents=True, exist_ok=True)

    for split, count in SPLITS.items():
        split_dir = output / split
        for idx in range(count):
            _create_example(idx, split, split_dir)

    _write_data_yaml(output)
    return output / "data.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic PCB dataset for demos")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/sample_dataset"),
        help="Directory where the dataset will be created",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not remove existing files before generating",
    )
    args = parser.parse_args()

    data_yaml = generate(args.output, overwrite=not args.no_overwrite)
    print(f"Sample dataset created! data.yaml = {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
