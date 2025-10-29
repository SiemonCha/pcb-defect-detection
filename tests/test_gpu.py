# test_gpu.py
import torch
import sys

print("="*60)
print("GPU DETECTION TEST")
print("="*60)

print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# Check CUDA (NVIDIA)
if torch.cuda.is_available():
    print("\n>>>>> CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    device = "cuda"

# Check MPS (Mac)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("\n>>>>> MPS available (Apple Silicon)")
    device = "mps"

# Check if torch was built with ROCm
elif torch.version.hip:
    print("\n>>>>> ROCm available (AMD)")
    print(f"   HIP version: {torch.version.hip}")
    device = "cuda"  # ROCm uses 'cuda' device string

else:
    print("\n----- No GPU detected, using CPU")
    device = "cpu"

# Test tensor creation
try:
    x = torch.rand(3, 3).to(device)
    print(f"\n>>>>> Tensor test passed on {device}")
    print(f"   Device: {x.device}")
except Exception as e:
    print(f"\n----- Tensor test failed: {e}")

print("\n" + "="*60)