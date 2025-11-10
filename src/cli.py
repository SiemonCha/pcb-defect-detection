"""
Unified command-line interface for PCB defect detection utilities.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Iterable, Optional


def _load_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(f"[ERROR] Missing dependency '{missing}'. Install the prerequisites listed in requirements.txt.")
        return None


def _run_cli(main_fn, args: Optional[Iterable[str]] = None) -> int:
    """Execute a module main() that accepts optional args iterable."""
    try:
        if args is None:
            result = main_fn()
        else:
            result = main_fn(args=args)
    except SystemExit as exc:  # module may call sys.exit
        return int(exc.code or 0)

    if result is None or result is True:
        return 0
    if isinstance(result, int):
        return result
    return 0


def cmd_dataset_analysis(_args: argparse.Namespace) -> int:
    dataset_module = _load_module("analysis.dataset")
    if not dataset_module:
        return 1

    return _run_cli(dataset_module.main)


def cmd_failure_analysis(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.top is not None:
        cli_args += ["--top", str(args.top)]
    if args.iou_threshold is not None:
        cli_args += ["--iou-threshold", str(args.iou_threshold)]
    failures_module = _load_module("analysis.failures")
    if not failures_module:
        return 1

    return _run_cli(failures_module.main, cli_args)


def cmd_interpretability(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.image:
        cli_args += ["--image", args.image]
    if args.samples is not None:
        cli_args += ["--samples", str(args.samples)]
    interpretability_module = _load_module("evaluation.interpretability")
    if not interpretability_module:
        return 1

    return _run_cli(interpretability_module.main, cli_args)


def cmd_quick_analysis(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.skip_failures:
        cli_args.append("--skip-failures")
    if args.skip_robustness:
        cli_args.append("--skip-robustness")
    if args.failure_top is not None:
        cli_args += ["--failure-top", str(args.failure_top)]
    if args.interpretability_samples is not None:
        cli_args += ["--interpretability-samples", str(args.interpretability_samples)]
    if args.robustness_samples is not None:
        cli_args += ["--robustness-samples", str(args.robustness_samples)]
    quick_suite = _load_module("analysis.quick_suite")
    if not quick_suite:
        return 1

    return _run_cli(quick_suite.main, cli_args)


def cmd_robustness(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.samples is not None:
        cli_args += ["--samples", str(args.samples)]
    if args.noise_levels:
        cli_args += ["--noise-levels", *map(str, args.noise_levels)]
    robustness_module = _load_module("evaluation.robustness")
    if not robustness_module:
        return 1

    return _run_cli(robustness_module.main, cli_args)


def cmd_check_dataset(args: argparse.Namespace) -> int:
    cli_args = ["--auto-fix"] if args.auto_fix else []
    check_dataset_module = _load_module("setup.check_dataset")
    if not check_dataset_module:
        return 1

    return _run_cli(check_dataset_module.main, cli_args)


def cmd_check_images(args: argparse.Namespace) -> int:
    cli_args = ["--fix"] if args.fix else []
    check_images_module = _load_module("setup.check_image_ranges")
    if not check_images_module:
        return 1

    return _run_cli(check_images_module.main, cli_args)


def cmd_verify_setup(_args: argparse.Namespace) -> int:
    verify_module = _load_module("setup.verify")
    if not verify_module:
        return 1

    return _run_cli(verify_module.main)


def cmd_data_download(_args: argparse.Namespace) -> int:
    data_download_module = _load_module("data.data_download")
    if not data_download_module:
        return 1

    return _run_cli(data_download_module.main)


def cmd_train_baseline(_args: argparse.Namespace) -> int:
    baseline_module = _load_module("training.baseline")
    if not baseline_module:
        return 1

    cli_args = []
    if hasattr(_args, "epochs") and _args.epochs is not None:
        cli_args += ["--epochs", str(_args.epochs)]
    if hasattr(_args, "imgsz") and _args.imgsz is not None:
        cli_args += ["--imgsz", str(_args.imgsz)]
    if hasattr(_args, "batch") and _args.batch is not None:
        cli_args += ["--batch", str(_args.batch)]
    if hasattr(_args, "patience") and _args.patience is not None:
        cli_args += ["--patience", str(_args.patience)]
    if getattr(_args, "model", None):
        cli_args += ["--model", _args.model]
    if getattr(_args, "project", None):
        cli_args += ["--project", _args.project]
    if getattr(_args, "name", None):
        cli_args += ["--name", _args.name]

    return _run_cli(baseline_module.main, cli_args)


def cmd_train_production(_args: argparse.Namespace) -> int:
    production_module = _load_module("training.production")
    if not production_module:
        return 1

    cli_args = []
    if hasattr(_args, "epochs") and _args.epochs is not None:
        cli_args += ["--epochs", str(_args.epochs)]
    if hasattr(_args, "imgsz") and _args.imgsz is not None:
        cli_args += ["--imgsz", str(_args.imgsz)]
    if hasattr(_args, "batch") and _args.batch is not None:
        cli_args += ["--batch", str(_args.batch)]
    if hasattr(_args, "patience") and _args.patience is not None:
        cli_args += ["--patience", str(_args.patience)]
    if getattr(_args, "model", None):
        cli_args += ["--model", _args.model]
    if getattr(_args, "project", None):
        cli_args += ["--project", _args.project]
    if getattr(_args, "name", None):
        cli_args += ["--name", _args.name]

    return _run_cli(production_module.main, cli_args)


def cmd_transfer_learning(args: argparse.Namespace) -> int:
    cli_args = ["--data", args.data]
    if args.base_model:
        cli_args += ["--base-model", args.base_model]
    if args.epochs is not None:
        cli_args += ["--epochs", str(args.epochs)]
    if args.batch is not None:
        cli_args += ["--batch", str(args.batch)]
    if args.lr is not None:
        cli_args += ["--lr", str(args.lr)]
    if args.freeze is not None:
        cli_args += ["--freeze", str(args.freeze)]
    if args.name:
        cli_args += ["--name", args.name]
    transfer_module = _load_module("training.transfer")
    if not transfer_module:
        return 1

    return _run_cli(transfer_module.main, cli_args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args.append(args.model)
    evaluate_module = _load_module("evaluation.evaluate")
    if not evaluate_module:
        return 1

    return _run_cli(evaluate_module.main, cli_args)


def cmd_confusion(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args.append(args.model)
    if getattr(args, "split", None):
        cli_args += ["--split", args.split]
    confusion_module = _load_module("evaluation.confusion")
    if not confusion_module:
        return 1

    return _run_cli(confusion_module.main, cli_args)


def cmd_export_onnx(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args.append(args.model)
    export_module = _load_module("deployment.export_onnx")
    if not export_module:
        return 1

    return _run_cli(export_module.main, cli_args)


def cmd_quantize(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.calibration_samples is not None:
        cli_args += ["--calibration-samples", str(args.calibration_samples)]
    if args.format:
        cli_args += ["--format", args.format]
    quantize_module = _load_module("deployment.quantize")
    if not quantize_module:
        return 1

    return _run_cli(quantize_module.main, cli_args)


def cmd_monitor(args: argparse.Namespace) -> int:
    cli_args = []
    if args.api_url:
        cli_args += ["--api-url", args.api_url]
    if args.analyze:
        cli_args += ["--analyze", args.analyze]
    if args.duration is not None:
        cli_args += ["--duration", str(args.duration)]
    if args.interval is not None:
        cli_args += ["--interval", str(args.interval)]
    monitoring_module = _load_module("deployment.monitoring")
    if not monitoring_module:
        return 1

    return _run_cli(monitoring_module.main, cli_args)


def cmd_api(args: argparse.Namespace) -> int:
    cli_args = []
    if args.model:
        cli_args += ["--model", args.model]
    if args.host:
        cli_args += ["--host", args.host]
    if args.port is not None:
        cli_args += ["--port", str(args.port)]
    api_module = _load_module("deployment.api")
    if not api_module:
        return 1

    return _run_cli(api_module.main, cli_args)


COMMANDS = {}


def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    def add_command(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(name, help=help_text)
        parser.set_defaults(func=handler)
        COMMANDS[name] = handler
        return parser

    add_command("dataset-analysis", cmd_dataset_analysis, "Analyze dataset class distribution and imbalance")

    parser_fail = add_command("failure-analysis", cmd_failure_analysis, "Visualize failure cases")
    parser_fail.add_argument("--model", type=str, help="Model weights path")
    parser_fail.add_argument("--top", type=int, default=20, help="Number of failures to visualize")
    parser_fail.add_argument("--iou-threshold", type=float, default=0.5, help="IoU matching threshold")

    parser_interp = add_command("interpretability", cmd_interpretability, "Generate attention maps")
    parser_interp.add_argument("--model", type=str, help="Model weights path")
    parser_interp.add_argument("--image", type=str, help="Specific image to visualize")
    parser_interp.add_argument("--samples", type=int, default=10, help="Random samples to visualize")

    parser_quick = add_command("quick-analysis", cmd_quick_analysis, "Run the quick analysis suite")
    parser_quick.add_argument("--model", type=str, help="Model weights path")
    parser_quick.add_argument("--skip-failures", action="store_true", help="Skip failure analysis")
    parser_quick.add_argument("--skip-robustness", action="store_true", help="Skip robustness testing")
    parser_quick.add_argument("--failure-top", type=int, default=20, help="Number of failure cases to visualize")
    parser_quick.add_argument("--interpretability-samples", type=int, default=10, help="Samples for attention maps")
    parser_quick.add_argument("--robustness-samples", type=int, default=10, help="Samples for robustness testing")

    parser_robust = add_command("robustness", cmd_robustness, "Run robustness evaluations")
    parser_robust.add_argument("--model", type=str, help="Model weights path")
    parser_robust.add_argument("--samples", type=int, default=10, help="Number of samples to test")
    parser_robust.add_argument("--noise-levels", type=float, nargs="*", help="Override noise levels")

    parser_check_ds = add_command("check-dataset", cmd_check_dataset, "Validate dataset annotations")
    parser_check_ds.add_argument("--auto-fix", action="store_true", help="Convert COCO segmentation to YOLO boxes")

    parser_check_img = add_command("check-images", cmd_check_images, "Validate image pixel ranges")
    parser_check_img.add_argument("--fix", action="store_true", help="Convert float images to uint8")

    add_command("verify-setup", cmd_verify_setup, "Verify required dependencies")
    add_command("data-download", cmd_data_download, "Download the dataset via configured provider")

    parser_train_base = add_command("train-baseline", cmd_train_baseline, "Train baseline YOLOv8 model")
    parser_train_base.add_argument("--epochs", type=int, help="Number of epochs")
    parser_train_base.add_argument("--imgsz", type=int, help="Image size")
    parser_train_base.add_argument("--batch", type=int, help="Batch size")
    parser_train_base.add_argument("--patience", type=int, help="Early stopping patience")
    parser_train_base.add_argument("--model", type=str, help="Pretrained weights path")
    parser_train_base.add_argument("--project", type=str, help="Training project directory")
    parser_train_base.add_argument("--name", type=str, help="Run name")

    parser_train_prod = add_command("train-production", cmd_train_production, "Train production YOLOv8 model")
    parser_train_prod.add_argument("--epochs", type=int, help="Number of epochs")
    parser_train_prod.add_argument("--imgsz", type=int, help="Image size")
    parser_train_prod.add_argument("--batch", type=int, help="Batch size override")
    parser_train_prod.add_argument("--patience", type=int, help="Early stopping patience")
    parser_train_prod.add_argument("--model", type=str, help="Pretrained weights path")
    parser_train_prod.add_argument("--project", type=str, help="Training project directory")
    parser_train_prod.add_argument("--name", type=str, help="Run name")

    parser_transfer = add_command("transfer-learning", cmd_transfer_learning, "Fine-tune model on new data")
    parser_transfer.add_argument("--data", type=str, required=True, help="Path to new dataset data.yaml")
    parser_transfer.add_argument("--base-model", type=str, help="Base weights to fine-tune")
    parser_transfer.add_argument("--epochs", type=int, help="Training epochs")
    parser_transfer.add_argument("--batch", type=int, help="Batch size")
    parser_transfer.add_argument("--lr", type=float, help="Learning rate")
    parser_transfer.add_argument("--freeze", type=int, help="Freeze backbone for N epochs")
    parser_transfer.add_argument("--name", type=str, help="Run name")

    parser_eval = add_command("evaluate", cmd_evaluate, "Evaluate a trained model")
    parser_eval.add_argument("model", nargs="?", help="Model weights path (auto-detect if omitted)")

    parser_conf = add_command("confusion", cmd_confusion, "Generate confusion matrix analysis")
    parser_conf.add_argument("model", nargs="?", help="Model weights path (auto-detect if omitted)")
    parser_conf.add_argument("--split", type=str, help="Dataset split to evaluate (default: test)")

    parser_export = add_command("export-onnx", cmd_export_onnx, "Export model to ONNX format")
    parser_export.add_argument("model", nargs="?", help="Model weights path (auto-detect if omitted)")

    parser_quant = add_command("quantize", cmd_quantize, "Quantize model for production")
    parser_quant.add_argument("--model", type=str, help="Model weights path")
    parser_quant.add_argument("--calibration-samples", type=int, help="Calibration sample count")
    parser_quant.add_argument("--format", type=str, choices=["onnx", "tflite", "edgetpu"], default="onnx")

    parser_monitor = add_command("monitor", cmd_monitor, "Monitor inference performance")
    parser_monitor.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    parser_monitor.add_argument("--analyze", type=str, help="Analyze existing log file")
    parser_monitor.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds")
    parser_monitor.add_argument("--interval", type=int, default=10, help="Interval between requests in seconds")

    parser_api = add_command("api", cmd_api, "Launch FastAPI inference server")
    parser_api.add_argument("--model", type=str, help="Model weights path")
    parser_api.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser_api.add_argument("--port", type=int, default=8000, help="Port number")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PCB Defect Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _register_commands(subparsers)
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

