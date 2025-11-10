"""Export trained YOLO models to ONNX and benchmark inference performance."""

from __future__ import annotations

import argparse
import glob
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from ultralytics import YOLO


def find_best_model() -> str:
    """Return the most recent trained model checkpoint."""
    patterns = [
        "runs/train/production_yolov8s*/weights/best.pt",
        "runs/train/baseline_yolov8n*/weights/best.pt",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    raise FileNotFoundError("No trained model found. Run: python -m cli train-baseline")


def benchmark_inference(model, runs: int = 100, force_cpu: bool = False) -> Optional[dict[str, float]]:
    """Benchmark inference by executing ``model`` repeatedly and timing each pass."""
    if force_cpu:
        dummy = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32) / 255.0
    else:
        if torch.cuda.is_available():
            dummy = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32).cuda() / 255.0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dummy = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32).to("mps") / 255.0
        else:
            dummy = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32) / 255.0

    successful = 0
    for _ in range(10):
        try:
            model(dummy, verbose=False)
            successful += 1
        except Exception:
            if successful == 0:
                return None
            break
    if successful == 0:
        return None

    timings: list[float] = []
    for _ in range(runs):
        try:
            start = time.perf_counter()
            model(dummy, verbose=False)
            timings.append((time.perf_counter() - start) * 1000)
        except Exception:
            continue
    if len(timings) < max(1, runs // 2):
        return None

    return {
        "mean": float(np.mean(timings)),
        "std": float(np.std(timings)),
        "min": float(np.min(timings)),
        "max": float(np.max(timings)),
    }


def ensure_onnx_dependencies(platform_name: str) -> tuple[bool, list[str]]:
    """Verify ONNX Runtime dependencies are available and return provider list."""
    try:
        import onnxruntime as ort
        import onnxslim  # noqa: F401
    except Exception:
        print("Required ONNX packages (onnxruntime, onnxslim) are missing.")
        print("Install the appropriate wheel for your platform, e.g.:")
        print("  pip install onnxruntime onnxslim")
        print("or GPU-specific builds such as onnxruntime-gpu / onnxruntime-rocm.")
        return False, []

    providers: list[str] = []
    try:
        providers = ort.get_available_providers()
        if platform_name.startswith("AMD") and not any(
            provider in providers for provider in ("ROCMExecutionProvider", "MIGraphXExecutionProvider")
        ):
            print("Warning: onnxruntime is present but no ROCm provider was loaded.")
            print("Available providers:", ", ".join(providers) or "<none>")
    except Exception as exc:
        print("Warning: unable to query ONNX Runtime providers:", exc)
    return True, providers


def format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name} Inference: {stats['mean']:.2f}ms +/- {stats['std']:.2f}ms\n"
        f"  Min: {stats['min']:.2f}ms | Max: {stats['max']:.2f}ms"
    )


def run_export(model_path: Optional[str], runs: int = 100) -> int:
    """Perform ONNX export and benchmarking for the supplied model path."""
    try:
        selected_model = Path(model_path) if model_path else Path(find_best_model())
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    if not selected_model.exists():
        print(f"ERROR: model not found at {selected_model}")
        return 1

    print("\n" + "=" * 60)
    print("PLATFORM DETECTION")
    print("=" * 60)

    if torch.cuda.is_available():
        if torch.version.hip:
            platform_name = f"AMD GPU (ROCm {torch.version.hip})"
            device_name = torch.cuda.get_device_name(0)
            print(f"Platform: {platform_name}")
            print(f"Device: {device_name}")
            print("Note: ONNX Runtime may fall back to CPU if ROCm providers are unavailable.")
        else:
            platform_name = f"NVIDIA GPU (CUDA {torch.version.cuda})"
            device_name = torch.cuda.get_device_name(0)
            print(f"Platform: {platform_name}")
            print(f"Device: {device_name}")
            print("ONNX Runtime can utilise CUDA execution providers.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        platform_name = "Apple Silicon (MPS)"
        print(f"Platform: {platform_name}")
        print("Note: ONNX Runtime currently uses CPU providers on Apple Silicon.")
    else:
        platform_name = "CPU"
        print("Platform: CPU")

    deps_ok, providers = ensure_onnx_dependencies(platform_name)
    if not deps_ok:
        return 1

    print(f"Loading model: {selected_model}")
    model = YOLO(str(selected_model))

    print("\nBenchmarking PyTorch model...")
    torch_stats = benchmark_inference(model, runs=runs)
    if torch_stats:
        print(format_stats("PyTorch", torch_stats))
    else:
        print("PyTorch benchmarking failed or produced insufficient samples.")

    print("\nExporting to ONNX...")
    onnx_path = Path(model.export(format="onnx", simplify=True))
    print(f"Exported ONNX model to {onnx_path}")

    print("\nLoading ONNX model via Ultralytics wrapper...")
    onnx_model = YOLO(str(onnx_path))

    print("\nBenchmarking ONNX model...")
    if providers:
        print("Available ONNX Runtime providers:", ", ".join(providers))
    onnx_stats = benchmark_inference(onnx_model, runs=runs)

    if onnx_stats is None:
        print("ONNX benchmarking via Ultralytics wrapper failed; attempting CPU fallback.")
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
            for _ in range(10):
                session.run(None, {session.get_inputs()[0].name: dummy})
            timings = []
            for _ in range(runs):
                start = time.perf_counter()
                session.run(None, {session.get_inputs()[0].name: dummy})
                timings.append((time.perf_counter() - start) * 1000)
            onnx_stats = {
                "mean": float(np.mean(timings)),
                "std": float(np.std(timings)),
                "min": float(np.min(timings)),
                "max": float(np.max(timings)),
            }
            print(format_stats("ONNX (CPU)", onnx_stats))
        except Exception as exc:
            print("CPU fallback failed:", exc)
            print("ONNX export succeeded but no benchmark numbers are available for this environment.")

    elif onnx_stats:
        print(format_stats("ONNX", onnx_stats))

    if torch_stats and onnx_stats:
        speedup = torch_stats["mean"] / onnx_stats["mean"]
        latency_delta = torch_stats["mean"] - onnx_stats["mean"]
        print("\n" + "=" * 60)
        print(f"Speedup: {speedup:.2f}x")
        print(f"Latency reduction: {latency_delta:.2f}ms")
        print("=" * 60)
    else:
        print("\nSkipped speedup calculation because one or both benchmarks were unavailable.")

    pytorch_size = selected_model.stat().st_size / (1024 ** 2)
    onnx_size = onnx_path.stat().st_size / (1024 ** 2)
    print("\nModel sizes:")
    print(f"  PyTorch: {pytorch_size:.2f} MB")
    print(f"  ONNX:    {onnx_size:.2f} MB")

    results_path = onnx_path.parent / "benchmark_results.txt"
    with results_path.open("w") as handle:
        handle.write("INFERENCE BENCHMARK RESULTS\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Model: {selected_model.name}\n\n")
        if torch_stats:
            handle.write(format_stats("PyTorch", torch_stats) + "\n\n")
        else:
            handle.write("PyTorch benchmark: unavailable\n\n")
        if onnx_stats:
            handle.write(format_stats("ONNX", onnx_stats) + "\n\n")
        else:
            handle.write("ONNX benchmark: unavailable\n\n")
        if torch_stats and onnx_stats:
            handle.write(f"Speedup: {torch_stats['mean'] / onnx_stats['mean']:.2f}x\n")
            handle.write(f"Latency reduction: {torch_stats['mean'] - onnx_stats['mean']:.2f}ms\n")
        else:
            handle.write("Speedup: N/A\n")
    print(f"\nSaved benchmark summary to {results_path}")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"onnx_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with log_file.open("w") as handle:
        handle.write("ONNX EXPORT BENCHMARK\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Timestamp: {datetime.now().isoformat()}\n")
        handle.write(f"Model: {selected_model.name}\n")
        handle.write(f"ONNX Path: {onnx_path}\n\n")
        if torch_stats:
            handle.write(format_stats("PyTorch", torch_stats) + "\n")
        else:
            handle.write("PyTorch benchmark: unavailable\n")
        if onnx_stats:
            handle.write(format_stats("ONNX", onnx_stats) + "\n")
        else:
            handle.write("ONNX benchmark: unavailable\n")
        handle.write(f"PyTorch size: {pytorch_size:.2f} MB\n")
        handle.write(f"ONNX size: {onnx_size:.2f} MB\n")
    print(f"Logged benchmark details to {log_file}")

    print("\nUse the ONNX model for deployment with:")
    print(f"  model = YOLO('{onnx_path}')")
    return 0


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a trained YOLO model to ONNX and benchmark it")
    parser.add_argument("model", nargs="?", help="Path to trained weights (defaults to newest run)")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed inference iterations")
    parsed = parser.parse_args(args=args)
    return run_export(parsed.model, runs=parsed.runs)


if __name__ == "__main__":
    raise SystemExit(main())


