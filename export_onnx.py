"""
Export trained model to ONNX format with speed benchmarking

Usage:
    python export_onnx.py                        # Auto-detect best model
    python export_onnx.py runs/.../best.pt       # Specific model
"""

from ultralytics import YOLO
import torch
import time
import os
import sys
import glob
import numpy as np

def find_best_model():
    """Find the best trained model"""
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    
    raise FileNotFoundError(
        "No trained model found. Train first:\n"
        "  python train_baseline.py"
    )

def benchmark_inference(model, format_name, runs=100, use_cpu=False):
    """Benchmark inference speed"""
    # Create input tensor and normalize to 0-1 range
    if use_cpu:
        dummy_input = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32) / 255.0
    else:
        # Use same device as model
        if torch.cuda.is_available():
            dummy_input = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32).cuda() / 255.0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dummy_input = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32).to('mps') / 255.0
        else:
            dummy_input = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.float32) / 255.0
    
    # Warmup
    successful_runs = 0
    for _ in range(10):
        try:
            model(dummy_input, verbose=False)
            successful_runs += 1
        except Exception as e:
            # If warmup fails, this format won't work
            if successful_runs == 0:
                return None
            break
    
    if successful_runs == 0:
        return None
    
    # Benchmark
    times = []
    for _ in range(runs):
        try:
            start = time.perf_counter()
            model(dummy_input, verbose=False)
            times.append((time.perf_counter() - start) * 1000)
        except Exception as e:
            # Skip failed runs
            continue
    
    if len(times) < runs // 2:  # If less than 50% succeeded
        return None
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Get model path
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = find_best_model()

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

# Detect platform
print("\n" + "="*60)
print("PLATFORM DETECTION")
print("="*60)

if torch.cuda.is_available():
    if torch.version.hip:
        platform_name = f"AMD GPU (ROCm {torch.version.hip})"
        device_name = torch.cuda.get_device_name(0)
        print(f"Platform: {platform_name}")
        print(f"Device: {device_name}")
        print("Note: ONNX Runtime may use CPU (ROCm support limited)")
    else:
        platform_name = f"NVIDIA GPU (CUDA {torch.version.cuda})"
        device_name = torch.cuda.get_device_name(0)
        print(f"Platform: {platform_name}")
        print(f"Device: {device_name}")
        print("Note: ONNX Runtime will use CUDA")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    platform_name = "Apple Silicon (MPS)"
    print(f"Platform: {platform_name}")
    print("Note: ONNX Runtime will use CPU (MPS not supported)")
else:
    platform_name = "CPU"
    print(f"Platform: {platform_name}")
    print("Note: ONNX Runtime will use CPU")

print(f"==== Loading model: {model_path}")
model = YOLO(model_path)

# Benchmark original PyTorch model
print("\n==== Benchmarking PyTorch model...")
pt_stats = benchmark_inference(model, "PyTorch")
if pt_stats is None:
    print('----- PyTorch benchmarking failed or incomplete. Skipping PyTorch stats.')
else:
    print(f"PyTorch Inference: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms")
    print(f"  Min: {pt_stats['min']:.2f}ms | Max: {pt_stats['max']:.2f}ms")

# Export to ONNX
print("\n==== Exporting to ONNX...")

# Ensure required ONNX packages are present before calling Ultralytics export.
# Ultralytics may attempt to auto-install 'onnxruntime' at runtime which can
# replace a ROCm-enabled wheel. To avoid that, require the user to install
# the correct wheel beforehand and fail fast with a clear message.
try:
    import onnxruntime as _ort  # noqa: F401
    import onnxslim as _os  # noqa: F401
except Exception:
    print("----- ERROR: Required ONNX packages (onnxruntime, onnxslim) are missing.")
    print("Install the ROCm-enabled ONNX Runtime in your environment and re-run:")
    print("  conda activate onnx311")
    print("  pip install --force-reinstall onnxruntime-rocm==1.22.2.post1")
    print("Or, if you don't have an AMD GPU, install the CPU package:")
    print("  pip install onnxruntime onnxslim")
    sys.exit(1)

# If on AMD, warn if ROCm provider isn't present in this python process
try:
    providers = _ort.get_available_providers()
    if platform_name.startswith('AMD') and 'ROCMExecutionProvider' not in providers and 'MIGraphXExecutionProvider' not in providers:
        print("----- WARNING: onnxruntime is installed but no ROCm provider was found in this process.")
        print("Providers available:", providers)
        print("If you expect ROCm support, ensure you installed a matching onnxruntime-rocm wheel and required system libs.")
except Exception:
    pass

onnx_path = model.export(format='onnx', simplify=True)
print(f"Exported to: {onnx_path}")

# Load ONNX model
print("\n==== Loading ONNX model...")
onnx_model = YOLO(onnx_path)

# Benchmark ONNX model
print("\n==== Benchmarking ONNX model...")

# Check what ONNX Runtime providers are available and pick the best one for the current platform
available_providers = []
try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print("==== Available ONNX providers:", ", ".join(available_providers))
except Exception as e:
    print("----- ONNX Runtime is not available or failed to import:", e)

onnx_stats = benchmark_inference(onnx_model, "ONNX")

if onnx_stats is None:
    print("\n----- ONNX GPU benchmarking failed (trying CPU fallback...)")
    
    # Try CPU fallback
    try:
        import onnxruntime as ort
        # Force CPU provider
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        print(">>>>> Using ONNX Runtime: CPUExecutionProvider")

        # Simple CPU benchmark
        dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

        # Warmup
        for _ in range(10):
            ort_session.run(None, {ort_session.get_inputs()[0].name: dummy})

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            ort_session.run(None, {ort_session.get_inputs()[0].name: dummy})
            times.append((time.perf_counter() - start) * 1000)

        onnx_stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

        print(f"ONNX Inference (CPU): {onnx_stats['mean']:.2f}ms ± {onnx_stats['std']:.2f}ms")
        print(f"  Min: {onnx_stats['min']:.2f}ms | Max: {onnx_stats['max']:.2f}ms")

    except Exception as e:
        print(f"----- CPU fallback also failed: {e}")
        print("\nNote: ONNX export successful but benchmarking failed.")
        print("This may happen on some GPU platforms (ROCm, MPS) if ONNX Runtime providers are not installed.")
        print(f"\n>>>>> ONNX model exported: {onnx_path}")
        print(">>>>> Model can still be used for deployment")

        # Save minimal report
        results_file = os.path.join(os.path.dirname(onnx_path), 'benchmark_results.txt')
        with open(results_file, 'w') as f:
            f.write("INFERENCE BENCHMARK RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {os.path.basename(model_path)}\n\n")
            f.write(f"Platform: {platform_name}\n\n")
            f.write(f"PyTorch Inference:\n")
            if pt_stats:
                f.write(f"  Mean: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms\n")
                f.write(f"  Range: [{pt_stats['min']:.2f}ms - {pt_stats['max']:.2f}ms]\n\n")
            f.write(f"ONNX Inference: Benchmarking failed (platform compatibility)\n")
            f.write(f"ONNX export successful: {onnx_path}\n")

        print(f"\n>>>>> Partial results saved to: {results_file}")
        sys.exit(0)
else:
    # Determine which provider was used
    print(">>>>> ONNX Runtime GPU acceleration detected")
    print(f"ONNX Inference: {onnx_stats['mean']:.2f}ms ± {onnx_stats['std']:.2f}ms")
    print(f"  Min: {onnx_stats['min']:.2f}ms | Max: {onnx_stats['max']:.2f}ms")

# Calculate speedup (only if both benchmarks succeeded)
if pt_stats is not None and onnx_stats is not None:
    speedup = pt_stats['mean'] / onnx_stats['mean']
    latency_diff = pt_stats['mean'] - onnx_stats['mean']

    print("\n" + "="*60)

    # Check if comparing GPU to CPU
    if speedup < 1.0:
        print("----- NOTE: PyTorch used GPU, ONNX used CPU")
        print(f">>>>> PyTorch (GPU): {pt_stats['mean']:.2f}ms")
        print(f">>>>> ONNX (CPU): {onnx_stats['mean']:.2f}ms")
        print(f">>>>> Result: CPU is {1/speedup:.2f}x slower than GPU (expected)")
        print("\nFor fair comparison both should use the same device. ONNX still provides deployment benefits.")
    else:
        print(f">>>>> SPEEDUP: {speedup:.2f}x faster")
        print(f">>>>> Latency reduction: {latency_diff:.2f}ms")

    print("="*60)
else:
    print('----- Skipping speedup calculation: missing benchmark results (PyTorch or ONNX)')

# Get file sizes
pt_size = os.path.getsize(model_path) / (1024**2)
onnx_size = os.path.getsize(onnx_path) / (1024**2)
print(f"\nModel sizes:")
print(f"  PyTorch: {pt_size:.2f} MB")
print(f"  ONNX:    {onnx_size:.2f} MB")

# Save benchmark results (original location)
results_file = os.path.join(os.path.dirname(onnx_path), 'benchmark_results.txt')
with open(results_file, 'w') as f:
    f.write("INFERENCE BENCHMARK RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {os.path.basename(model_path)}\n\n")

    # PyTorch stats (if present)
    if pt_stats is not None:
        f.write(f"PyTorch Inference:\n")
        f.write(f"  Mean: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms\n")
        f.write(f"  Range: [{pt_stats['min']:.2f}ms - {pt_stats['max']:.2f}ms]\n\n")
    else:
        f.write("PyTorch Inference: Benchmarking failed or unavailable\n\n")

    # ONNX stats (if present)
    if onnx_stats is not None:
        f.write(f"ONNX Inference:\n")
        f.write(f"  Mean: {onnx_stats['mean']:.2f}ms ± {onnx_stats['std']:.2f}ms\n")
        f.write(f"  Range: [{onnx_stats['min']:.2f}ms - {onnx_stats['max']:.2f}ms]\n\n")
    else:
        f.write("ONNX Inference: Benchmarking failed or unavailable\n\n")

    # Speedup (only if both present)
    if pt_stats is not None and onnx_stats is not None:
        speedup = pt_stats['mean'] / onnx_stats['mean']
        latency_red = pt_stats['mean'] - onnx_stats['mean']
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Latency Reduction: {latency_red:.2f}ms\n")
    else:
        f.write("Speedup: N/A (missing benchmark results)\n")

print(f"\n>>>>> Results saved to: {results_file}")

# Also save to logs directory
from pathlib import Path
from datetime import datetime

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"onnx_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, 'w') as f:
    f.write("ONNX EXPORT BENCHMARK\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"ONNX Path: {onnx_path}\n\n")

    if pt_stats is not None:
        f.write(f"PyTorch Inference: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms\n")
    else:
        f.write("PyTorch Inference: unavailable\n")

    if onnx_stats is not None:
        f.write(f"ONNX Inference: {onnx_stats['mean']:.2f}ms ± {onnx_stats['std']:.2f}ms\n")
    else:
        f.write("ONNX Inference: unavailable\n")

    if pt_stats is not None and onnx_stats is not None:
        speedup = pt_stats['mean'] / onnx_stats['mean']
        latency_red = pt_stats['mean'] - onnx_stats['mean']
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Latency Reduction: {latency_red:.2f}ms\n\n")
    else:
        f.write("Speedup: N/A (missing benchmark results)\n\n")

    f.write(f"File Sizes:\n")
    f.write(f"  PyTorch: {pt_size:.2f} MB\n")
    f.write(f"  ONNX: {onnx_size:.2f} MB\n")

print(f">>>>> Benchmark also logged to: {log_file}")
print(f"\n>>>>> Use ONNX model for deployment:")
print(f"   model = YOLO('{onnx_path}')")


# Also save to logs directory
from pathlib import Path
from datetime import datetime

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"onnx_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_file, 'w') as f:
    f.write("ONNX EXPORT BENCHMARK\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"ONNX Path: {onnx_path}\n\n")

    if pt_stats is not None:
        f.write(f"PyTorch Inference: {pt_stats['mean']:.2f}ms ± {pt_stats['std']:.2f}ms\n")
    else:
        f.write("PyTorch Inference: unavailable\n")

    if onnx_stats is not None:
        f.write(f"ONNX Inference: {onnx_stats['mean']:.2f}ms ± {onnx_stats['std']:.2f}ms\n")
    else:
        f.write("ONNX Inference: unavailable\n")

    if pt_stats is not None and onnx_stats is not None:
        speedup = pt_stats['mean'] / onnx_stats['mean']
        latency_red = pt_stats['mean'] - onnx_stats['mean']
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Latency Reduction: {latency_red:.2f}ms\n\n")
    else:
        f.write("Speedup: N/A (missing benchmark results)\n\n")

    f.write(f"File Sizes:\n")
    f.write(f"  PyTorch: {pt_size:.2f} MB\n")
    f.write(f"  ONNX: {onnx_size:.2f} MB\n")

print(f">>>>> Benchmark also logged to: {log_file}")
print(f"\n>>>>> Use ONNX model for deployment:")
print(f"   model = YOLO('{onnx_path}')")


print(f">>>>> Benchmark also logged to: {log_file}")
print(f"\n>>>>> Use ONNX model for deployment:")
print(f"   model = YOLO('{onnx_path}')")