"""
YOLOv8s Production - Better accuracy with optimizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import argparse
import glob
import logging
import os
import platform
import time

os.environ.setdefault("ULTRALYTICS_MINIMAL", "True")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from ultralytics import YOLO
try:
    from ultralytics.yolo.utils import LOGGER
except ImportError:
    LOGGER = logging.getLogger("ultralytics")

LOGGER.setLevel(logging.ERROR)


def get_device_info() -> Tuple[str, bool]:
    """Get device information and capabilities."""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        if torch.version.hip is not None:
            print(f">>>> Training on: AMD GPU (ROCm) - {device_name}")
            return device, True

        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f">>>> Training on: NVIDIA GPU - {device_name} ({memory:.1f} GB)")
        return device, False

    if torch.backends.mps.is_available():
        device = "mps"
        print(f">>>> Training on: Apple Silicon - {platform.processor()}")
        return device, False

    device = "cpu"
    print(f">>>> Training on: CPU - {platform.processor()}")
    return device, False


def find_data_yaml() -> str:
    """Locate the project data.yaml describing the dataset."""
    dataset_path_file = Path("dataset_path.txt")
    if dataset_path_file.exists():
        dataset_path = dataset_path_file.read_text().strip()
        data_yaml = Path(dataset_path) / "data.yaml"
        if data_yaml.exists():
            return str(data_yaml)

    patterns = [
        "data/*/data.yaml",
        "data/data.yaml",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    raise FileNotFoundError("data.yaml not found. Run: python -m cli data-download")


def _register_epoch_logger(model: YOLO, total_epochs: int) -> None:
    def _on_epoch_end(trainer) -> None:
        epoch = getattr(trainer, "epoch", 0) + 1
        loss_items = getattr(trainer, "loss_items", None)
        message = f"Epoch {epoch}/{total_epochs} completed"
        if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
            message += (
                f" | box={loss_items[0]:.4f}"
                f" cls={loss_items[1]:.4f}"
                f" dfl={loss_items[2]:.4f}"
            )
        print(message, flush=True)

    model.add_callback("on_train_epoch_end", _on_epoch_end)


def _register_interval_logger(model: YOLO, total_epochs: int, interval_secs: int = 10) -> None:
    state = {"last_ts": time.time(), "last_batch": -1}

    def _on_batch_end(trainer) -> None:
        now = time.time()
        if now - state["last_ts"] < interval_secs:
            return

        epoch = getattr(trainer, "epoch", 0) + 1
        batch_idx = getattr(trainer, "batch_i", 0)
        total_batches = max(1, getattr(trainer, "nb", 0))

        diff_batches = max(0, batch_idx - state["last_batch"]) if state["last_batch"] >= 0 else 0
        elapsed = now - state["last_ts"]
        rate = diff_batches / elapsed if elapsed > 0 else 0.0

        progress = min(batch_idx + 1, total_batches) / total_batches * 100.0
        loss_items = getattr(trainer, "loss_items", None)

        bar_width = 12
        filled = int(progress / 100 * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)

        message = (
            f"{progress:5.1f}% {bar} "
            f"{batch_idx + 1}/{total_batches}"
        )
        if rate > 0:
            message += f" {rate*total_batches/60:.2f} it/min"
        if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
            message += (
                f" | box={loss_items[0]:.4f}"
                f" cls={loss_items[1]:.4f}"
                f" dfl={loss_items[2]:.4f}"
            )
        print(message, flush=True)

        state["last_ts"] = now
        state["last_batch"] = batch_idx

    model.add_callback("on_train_batch_end", _on_batch_end)


def train_production(
    *,
    epochs: int = 100,
    imgsz: int = 640,
    batch: Optional[int] = None,
    patience: int = 15,
    model_path: str = "yolov8s.pt",
    project: str = "runs/train",
    name: str = "production_yolov8s",
):
    """Train the YOLOv8s production model."""
    device, is_rocm = get_device_info()
    data_yaml = find_data_yaml()
    print(f"==== Using dataset: {data_yaml}")

    print("==== Loading YOLOv8s...")
    model = YOLO(model_path)
    _register_epoch_logger(model, epochs)
    _register_interval_logger(model, epochs)

    print("==== Starting production training...")
    effective_batch = batch if batch is not None else (16 if device == "cuda" else 8)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=effective_batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        plots=True,
        verbose=False,
        show=False,
        cache="ram" if device != "cpu" else False,
        amp=bool(device == "cuda" and not is_rocm),
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
    )

    print("\n>>>> Production training complete!")
    print(f">>>> Results: {project}/{name}")
    print(f">>>> Best: {project}/{name}/weights/best.pt")
    print("\n>>>> Next step: python -m cli evaluate")
    return results


def main(args: Optional[Iterable[str]] = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the YOLOv8s production model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="production_yolov8s")
    parsed = parser.parse_args(args=args)
    try:
        train_production(
            epochs=parsed.epochs,
            imgsz=parsed.imgsz,
            batch=parsed.batch,
            patience=parsed.patience,
            model_path=parsed.model,
            project=parsed.project,
            name=parsed.name,
        )
    except FileNotFoundError as err:
        print(f"xxxx {err}")
        raise


if __name__ == "__main__":
    main()