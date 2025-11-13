from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _load_session(model_path: Path, providers: Optional[list[str]] = None):
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "onnxruntime is required for benchmarking. Install onnxruntime to use this command."
        ) from exc

    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)


def benchmark_onnx_model(
    model_path: Path,
    *,
    runs: int = 100,
    warmup: int = 10,
    imgsz: int = 640,
    providers: Optional[list[str]] = None,
) -> dict[str, float]:
    session = _load_session(model_path, providers)
    input_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape

    batch = shape[0] if isinstance(shape[0], int) else 1
    data = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(max(0, warmup)):
        session.run(None, {input_name: data})

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: data})
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "runs": runs,
        "mean_ms": statistics.mean(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": np.percentile(timings, 95).item(),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "throughput_fps": (batch * 1000.0) / statistics.mean(timings),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference latency")
    parser.add_argument("model", type=Path, help="Path to ONNX model")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        help="Custom execution providers (fallbacks to default ordering)",
    )
    args = parser.parse_args(argv)

    stats = benchmark_onnx_model(
        args.model,
        runs=args.runs,
        warmup=args.warmup,
        imgsz=args.imgsz,
        providers=args.providers,
    )

    print("ONNX Benchmark Results")
    print("----------------------")
    print(f"Model: {args.model}")
    print(f"Runs: {stats['runs']}")
    print(f"Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"Median latency: {stats['p50_ms']:.2f} ms")
    print(f"P95 latency: {stats['p95_ms']:.2f} ms")
    print(f"Min latency: {stats['min_ms']:.2f} ms")
    print(f"Max latency: {stats['max_ms']:.2f} ms")
    print(f"Throughput: {stats['throughput_fps']:.2f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
