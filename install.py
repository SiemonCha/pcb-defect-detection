#!/usr/bin/env python3
"""
Auto-installer for PCB Defect Detection

Detects your platform and installs appropriate PyTorch version + dependencies.

Usage:
    python install.py
"""

import platform
import subprocess
import sys
import os
import glob
from pathlib import Path

def get_gpu_type():
    """Detect GPU type: NVIDIA, AMD, or None"""
    try:
        # Check for NVIDIA GPU
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return 'nvidia'
    except:
        try:
            # Check for AMD GPU with ROCm
            subprocess.run(['rocm-smi'], capture_output=True, check=True)
            return 'amd'
        except:
            return None

def build_onnxruntime_rocm():
    """Build ONNX Runtime from source with ROCm support"""
    print("\n" + "="*60)
    print("Building ONNX Runtime with ROCm support...")
    print("="*60)
    
    # Install build dependencies
    print("Installing build dependencies...")
    subprocess.run(
        "sudo apt update && sudo apt install -y cmake build-essential python3-dev python3-pip git ninja-build",
        shell=True,
        check=True
    )
    
    # Clone ONNX Runtime
    if not os.path.exists("onnxruntime"):
        subprocess.run(
            ["git", "clone", "--recursive", "https://github.com/microsoft/onnxruntime.git"],
            check=True
        )
    
    # Build ONNX Runtime
    build_cmd = [
        "./build.sh",
        "--config", "Release",
        "--build_wheel",
        "--use_rocm",
        "--rocm_home", "/opt/rocm"
    ]
    
    subprocess.run(build_cmd, cwd="onnxruntime", check=True)
    
    # Find and install the wheel
    import glob
    wheels = glob.glob("onnxruntime/build/Linux/Release/dist/onnxruntime_rocm*.whl")
    if wheels:
        wheel_path = wheels[0]
        subprocess.run([sys.executable, "-m", "pip", "install", wheel_path], check=True)
        print(f"✅ Installed ONNX Runtime wheel: {wheel_path}")
    else:
        print("❌ No ONNX Runtime wheel found after build")

def install_requirements():
    """Install dependencies based on platform"""

    print("="*60)
    print("PCB DEFECT DETECTION - AUTO INSTALLER")
    print("="*60)
    
    # Check Python version
    py_version = tuple(map(int, platform.python_version().split('.')))
    if py_version[0] != 3 or py_version[1] != 11:
        print("❌ Error: Python 3.11.x is required")
        print(f"Current version: {platform.python_version()}")
        print("\nPlease create a new conda environment with Python 3.11:")
        print("  conda create -n pcb311 python=3.11")
        print("  conda activate pcb311")
        print("Then run this installer again.")
        return False
    
    # Detect system
    system = platform.system().lower()
    gpu_type = get_gpu_type()
    
    print(f"\nDetected System: {system}")
    print(f"Detected GPU: {gpu_type if gpu_type else 'None (CPU only)'}")
    
    # Install common requirements first
    print("\n" + "="*60)
    print("Installing common requirements...")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Error installing requirements:")
        print(result.stderr)
        return False
    
    print("✅ Common requirements installed")
    
    # Install platform-specific PyTorch
    print("\n" + "="*60)
    print("Installing PyTorch for your platform...")
    print("="*60)
    
    if system == 'darwin' and platform.processor() == 'arm':
        print("Installing PyTorch for Apple Silicon (MPS)...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
    
    elif gpu_type == 'nvidia':
        print("Installing PyTorch for NVIDIA GPU (CUDA 11.8)...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ]
    
    elif gpu_type == 'amd':
        print("Installing PyTorch for AMD GPU (ROCm 6.0)...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--pre', 'torch', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/nightly/rocm6.0'
        ]
    
    else:
        print("Installing PyTorch for CPU...")
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision'
        ]
    
    result = subprocess.run(torch_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Error installing PyTorch:")
        print(result.stderr)
        return False
    
    print("✅ PyTorch installed")
    
    # Install ONNX Runtime
    print("\n" + "="*60)
    print("Installing ONNX Runtime...")
    print("="*60)
    
    if gpu_type == 'nvidia':
        print("Installing ONNX Runtime with CUDA support...")
        onnx_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'onnxruntime-gpu'
        ]
        subprocess.run(onnx_cmd, check=True)
    
    elif gpu_type == 'amd':
        print("Installing ONNX Runtime with ROCm support...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'onnxruntime-rocm==1.22.2.post1'
        ], check=True)
    
    else:
        print("Installing CPU ONNX Runtime...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'onnxruntime'
        ], check=True)
    
    print("✅ ONNX Runtime installed")
    
    # Verify installation
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    try:
        # Check PyTorch
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✅ MPS available (Apple Silicon)")
        elif hasattr(torch, 'version.hip') and torch.cuda.is_available():
            print(f"✅ ROCm available (AMD GPU): {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️  CPU only (no GPU detected)")
        
        # Check ONNX Runtime
        import onnxruntime as ort
        print(f"\n✅ ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download dataset: python data_download.py")
    print("  2. Train model: python train_baseline.py")
    print("  3. Analyze results: python auto_analyze.py")
    print("\nNote: auto_analyze.py runs evaluation/confusion matrix/ONNX export.")
    print("      It does NOT retrain the model.")
    
    return True

if __name__ == '__main__':
    success = install_requirements()
    sys.exit(0 if success else 1)