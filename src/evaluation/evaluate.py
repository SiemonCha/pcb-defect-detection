"""
Evaluate trained model on test set with automatic logging

Usage:
    python evaluate.py                           # Auto-detect best model
    python evaluate.py runs/.../weights/best.pt  # Specify model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ultralytics import YOLO
import os
import sys
import glob
import json
from datetime import datetime
from pathlib import Path
import yaml

from data import resolve_dataset_yaml


DEFAULT_MODEL_PATTERNS = [
    "runs/train/production_yolov8s*/weights/best.pt",
    "runs/train/baseline_yolov8n*/weights/best.pt",
]


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    speed: Dict[str, float]
    total_time_ms: float
    report_file: Path
    log_file: Path


def find_model_path(cli_arg: Optional[str] = None, patterns: Optional[List[str]] = None) -> str:
    """Locate a trained model to evaluate."""
    if cli_arg:
        if not os.path.exists(cli_arg):
            raise FileNotFoundError(f"Model not found: {cli_arg}")
        return cli_arg

    for pattern in patterns or DEFAULT_MODEL_PATTERNS:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)

    raise FileNotFoundError(
        "No trained model found. Run: python -m cli train-baseline\n"
    )


def find_data_yaml() -> Path:
    """Find dataset configuration."""
    return resolve_dataset_yaml()


def evaluate_model(model_path: str | Path, data_yaml: Path, log_dir: Path | None = None) -> EvaluationResult:
    """Run evaluation on the test split and persist outputs."""
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(exist_ok=True)

    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    model = YOLO(str(model_path))
    print(f">>>>> Evaluating: {model_path}")
    print(f">>>>> Dataset: {data_yaml}")

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        plots=True,
        save_json=True,
    )

    class_names = _normalise_class_names(metrics, data_yaml)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "dataset": str(data_yaml),
        "metrics": {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": {},
        "speed": {},
    }

    if class_names and hasattr(metrics.box, "ap50"):
        for name, ap in zip(class_names, metrics.box.ap50):
            results["per_class"][name] = {"AP@0.5": float(ap)}

    total_time = 0.0
    if hasattr(metrics, "speed") and metrics.speed:
        for key, val in metrics.speed.items():
            if isinstance(val, (int, float)):
                results["speed"][key] = float(val)
                total_time += float(val)
        if total_time:
            results["speed"]["total"] = total_time

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    if results["per_class"]:
        print("\n>>>>> Per-Class Performance:")
        print("-" * 60)
        for name, perf in results["per_class"].items():
            print(f"{name:20s} | AP@0.5: {perf['AP@0.5']:.4f}")

    if results["speed"]:
        print("\nInference speed:")
        for key, val in results["speed"].items():
            if key == "total":
                continue
            print(f"{key.capitalize():12s}: {val:.1f}ms")
        print(f"{'Total':12s}: {total_time:.1f}ms")

    json_log = log_dir / "evaluation_log.json"
    if json_log.exists():
        with json_log.open("r", encoding="utf-8") as file:
            all_results = json.load(file)
    else:
        all_results = []

    all_results.append(results)
    with json_log.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\n>>>>> Results logged to: {json_log}")

    report_file = log_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with report_file.open("w", encoding="utf-8") as file:
        file.write("EVALUATION REPORT\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Timestamp: {results['timestamp']}\n")
        file.write(f"Model: {results['model_name']}\n")
        file.write(f"Dataset: {data_yaml}\n\n")

        file.write("Overall Metrics:\n")
        file.write("-" * 60 + "\n")
        file.write(f"mAP@0.5:      {metrics.box.map50:.4f}\n")
        file.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        file.write(f"Precision:    {metrics.box.mp:.4f}\n")
        file.write(f"Recall:       {metrics.box.mr:.4f}\n\n")

        if results["per_class"]:
            file.write("Per-Class Performance:\n")
            file.write("-" * 60 + "\n")
            for name, perf in results["per_class"].items():
                file.write(f"{name:20s} | AP@0.5: {perf['AP@0.5']:.4f}\n")
            file.write("\n")

        if results["speed"]:
            file.write("Inference Speed:\n")
            file.write("-" * 60 + "\n")
            for key, val in results["speed"].items():
                if key == "total":
                    continue
                file.write(f"{key.capitalize():12s}: {val:.1f}ms\n")
            if total_time:
                file.write(f"{'Total':12s}: {total_time:.1f}ms\n")
            file.write("\n")

        file.write("Performance Assessment:\n")
        file.write("-" * 60 + "\n")

        if metrics.box.map50 > 0.85:
            file.write("PASS: Model meets 85% mAP@0.5 target\n")
        else:
            file.write(f"WARNING: mAP@0.5 is {metrics.box.map50:.1%}, target is 85%\n")
            file.write("   Recommendation: train YOLOv8m or increase epochs\n")

        if total_time and total_time < 100:
            file.write("PASS: Inference < 100ms target\n")
        elif total_time:
            file.write(f"WARNING: Inference is {total_time:.0f}ms\n")
            file.write("   Recommendation: use YOLOv8n or optimise to ONNX\n")

    print(f">>>>> Report saved to: {report_file}")

    print("\n" + "=" * 60)
    if metrics.box.map50 > 0.85:
        print(">>>>> PASS: Model meets 85% mAP@0.5 target")
    else:
        print(f">>>>> WARNING: mAP@0.5 is {metrics.box.map50:.1%}, target is 85%")
        print("   - Consider training YOLOv8m or increasing epochs")

    if total_time and total_time < 100:
        print(">>>>> PASS: Inference < 100ms target")
    elif total_time:
        print(f">>>>> WARNING: Inference is {total_time:.0f}ms")
        print("   - Consider using YOLOv8n or optimising to ONNX")

    print("\n>>>>> All results automatically logged to logs/ directory")
    print(f">>>>> View logs at: {log_dir.absolute()}")

    return EvaluationResult(
        metrics=results["metrics"],
        per_class=results["per_class"],
        speed=results["speed"],
        total_time_ms=total_time,
        report_file=report_file,
        log_file=json_log,
    )


def main(cli_args: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    argv = cli_args or sys.argv[1:]
    model_arg = argv[0] if argv else None
    data_yaml = find_data_yaml()
    model_path = find_model_path(model_arg)
    evaluate_model(model_path, data_yaml, Path("logs"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)


def _normalise_class_names(metrics, data_yaml: Path) -> list[str]:
    if hasattr(metrics, "names") and metrics.names:
        names = metrics.names
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        return list(names)

    with data_yaml.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    names_cfg = cfg.get("names", [])
    if isinstance(names_cfg, dict):
        return [names_cfg[k] for k in sorted(names_cfg)]
    return list(names_cfg)