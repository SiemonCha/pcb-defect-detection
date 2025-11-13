"""
YOLOv8n Baseline - Fast training with multi-platform support
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
from datetime import datetime

os.environ.setdefault("ULTRALYTICS_MINIMAL", "True")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from ultralytics import YOLO
try:
    from ultralytics.yolo.utils import LOGGER
except ImportError:
    LOGGER = logging.getLogger("ultralytics")

LOGGER.setLevel(logging.ERROR)

from config import load_config
from data import resolve_dataset_yaml
from tracking import configure_wandb, finish_wandb_run


CONFIG = load_config()
DATA_CFG = CONFIG.get("data", {})
BASELINE_CFG = CONFIG.get("training", {}).get("baseline", {})
TRACKING_CFG = CONFIG.get("tracking", {})


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


def find_data_yaml() -> Path:
    """Locate the project data.yaml describing the dataset."""
    return resolve_dataset_yaml()


def _register_epoch_logger(model: YOLO, total_epochs: int) -> None:
    """Emit a summary at the end of each epoch."""

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
    """Log progress every ~interval_secs seconds."""

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


def train_baseline(
    *,
    epochs: int = BASELINE_CFG.get("epochs", 50),
    imgsz: int = BASELINE_CFG.get("imgsz", 640),
    batch: int = BASELINE_CFG.get("batch", 8),
    patience: int = BASELINE_CFG.get("patience", 10),
    model_path: str = "yolov8n.pt",
    project: str = "runs/train",
    name: str = "baseline_yolov8n",
):
    """Train the YOLOv8n baseline model."""
    device, is_rocm = get_device_info()
    data_yaml_path = find_data_yaml()
    print(f"==== Using dataset: {data_yaml_path}")

    tracking_run = None
    run_name = name
    tracking_enabled = bool(TRACKING_CFG.get("enabled"))
    if tracking_enabled:
        prefix = TRACKING_CFG.get("run_name_prefix", "pcbdd")
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{prefix}-baseline-{timestamp}"
        tracking_run = configure_wandb(
            TRACKING_CFG,
            run_name,
            config={
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "patience": patience,
                "model_path": model_path,
                "dataset": str(data_yaml_path),
            },
            tags=["baseline"],
        )

    print("==== Training configuration ====")
    print(f"epochs={epochs} | imgsz={imgsz} | batch={batch} | patience={patience}")
    print(f"tracking={'enabled' if tracking_enabled else 'disabled'} | project={project} | name={name}")

    print("==== Loading YOLOv8n...")
    model = YOLO(model_path)
    _register_epoch_logger(model, epochs)
    _register_interval_logger(model, epochs)

    print("==== Starting training...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
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
    )

    if tracking_run is not None:
        summary = {}
        if hasattr(results, "results_dict"):
            summary = getattr(results, "results_dict") or {}
        elif isinstance(results, dict):
            summary = results
        if summary:
            try:
                tracking_run.log(summary)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to log summary metrics to W&B: {exc}")
        finish_wandb_run(tracking_run)

    print("\n>>>> Training complete!")
    print(f">>>> Results: {project}/{name}")
    print(f">>>> Best: {project}/{name}/weights/best.pt")
    if summary := getattr(results, "results_dict", None):
        print(">>>> Key metrics:")
        for key in ("metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"):
            if key in summary:
                print(f"     {key}: {summary[key]:.4f}")
    print("\n>>>> Next step: python -m cli evaluate")
    return results


def main(args: Optional[Iterable[str]] = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the YOLOv8n baseline model")
    parser.add_argument("--epochs", type=int, default=BASELINE_CFG.get("epochs", 50))
    parser.add_argument("--imgsz", type=int, default=BASELINE_CFG.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=BASELINE_CFG.get("batch", 8))
    parser.add_argument("--patience", type=int, default=BASELINE_CFG.get("patience", 10))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="baseline_yolov8n")
    parsed = parser.parse_args(args=args)
    try:
        train_baseline(
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