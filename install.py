#!/usr/bin/env python3
"""
Universal Auto-Installer for PCB Defect Detection

Detects your platform and installs appropriate packages:
- NVIDIA GPU: PyTorch CUDA 11.8 + ONNX Runtime GPU
- AMD GPU: PyTorch ROCm 6.0 + ONNX Runtime ROCm  
- Apple Silicon: PyTorch MPS + ONNX Runtime CPU
- CPU-only: PyTorch CPU + ONNX Runtime CPU

Usage:
    python install.py
    python install.py --force-cpu    # Force CPU installation
"""

import platform
import subprocess
import sys
import os
import argparse

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nxxxx Failed: {e}")
        return False

def get_gpu_type():
    """Detect GPU type: nvidia, amd, or None"""
    # Check NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return 'nvidia'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check AMD
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, check=True)
        return 'amd'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check AMD alternative
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, check=True)
        return 'amd'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return None

def check_apple_silicon():
    """Check if running on Apple Silicon"""
    if platform.system() == 'Darwin':
        proc = platform.processor()
        machine = platform.machine()
        # Apple Silicon reports 'arm' or 'arm64'
        return 'arm' in proc.lower() or machine == 'arm64'
    return False

def check_python_version():
    """Check Python version"""
    py_ver = platform.python_version()
    major, minor = map(int, py_ver.split('.')[:2])
    
    print(f"\n{'='*60}")
    print("Python Version Check")
    print(f"{'='*60}")
    print(f"Detected: Python {py_ver}")
    
    if major != 3:
        print(f"❌ Error: Python 3.x required, found Python {major}.x")
        return False
    
    if minor == 11:
        print(f"✅ Python 3.11.x - Perfect!")
        return True
    elif minor == 12:
        print(f"⚠️  Python 3.12 detected - should work but 3.11 is recommended")
        return True
    elif minor >= 10:
        print(f"⚠️  Python 3.{minor} detected - should work but 3.11 is recommended")
        return True
    else:
        print(f"❌ Error: Python 3.10+ required, found 3.{minor}")
        print("\nCreate a new environment:")
        print("  conda create -n pcb311 python=3.11")
        print("  conda activate pcb311")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install PCB Defect Detection dependencies")
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU-only installation')
    parser.add_argument('--skip-torch', action='store_true', help='Skip PyTorch installation')
    parser.add_argument('--skip-onnx', action='store_true', help='Skip ONNX Runtime installation')
    args = parser.parse_args()
    
    print("="*60)
    print("PCB DEFECT DETECTION - UNIVERSAL INSTALLER")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect platform
    print(f"\n{'='*60}")
    print("Platform Detection")
    print(f"{'='*60}")
    
    system = platform.system()
    gpu_type = None if args.force_cpu else get_gpu_type()
    is_apple_silicon = check_apple_silicon()
    
    print(f"Operating System: {system}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Determine installation type
    if args.force_cpu:
        install_type = 'cpu'
        print(f"\n====  Installation Mode: CPU-only (forced)")
    elif gpu_type == 'nvidia':
        install_type = 'nvidia'
        print(f"\n==== GPU Detected: NVIDIA")
        print(f"   Installation Mode: CUDA 11.8")
    elif gpu_type == 'amd':
        install_type = 'amd'
        print(f"\n==== GPU Detected: AMD")
        print(f"   Installation Mode: ROCm 6.0")
    elif is_apple_silicon:
        install_type = 'apple'
        print(f"\n>>>> Platform: Apple Silicon")
        print(f"   Installation Mode: MPS")
    else:
        install_type = 'cpu'
        print(f"\n====  Installation Mode: CPU-only")
    
    # Step 1: Install common requirements
    print(f"\n{'='*60}")
    print("Step 1/3: Installing Common Dependencies")
    print(f"{'='*60}")
    
    if not run_command(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        "Installing common packages from requirements.txt"
    ):
        print("\nxxxx Failed to install common dependencies")
        print("Check requirements.txt exists and is valid")
        sys.exit(1)

    print("\n>>>> Common dependencies installed")
    
    # Step 2: Install PyTorch
    if not args.skip_torch:
        print(f"\n{'='*60}")
        print("Step 2/3: Installing PyTorch")
        print(f"{'='*60}")
        
        if install_type == 'nvidia':
            print("Installing PyTorch with CUDA 11.8 support...")
            torch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ]
        
        elif install_type == 'amd':
            print("Installing PyTorch with ROCm 6.0 support...")
            torch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                '--pre', 'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/nightly/rocm6.0'
            ]
        
        elif install_type == 'apple':
            print("Installing PyTorch with MPS support (Apple Silicon)...")
            torch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio'
            ]
        
        else:  # cpu
            print("Installing PyTorch (CPU-only)...")
            torch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ]
        
        if not run_command(torch_cmd, "Installing PyTorch"):
            print("\n❌ Failed to install PyTorch")
            print("\nTry manual installation:")
            print(f"  {' '.join(torch_cmd)}")
            sys.exit(1)
        
        print("\n>>>> PyTorch installed")
    else:
        print("\n----  Skipping PyTorch installation (--skip-torch)")
    
    # Step 3: Install ONNX Runtime
    if not args.skip_onnx:
        print(f"\n{'='*60}")
        print("Step 3/3: Installing ONNX Runtime")
        print(f"{'='*60}")
        
        if install_type == 'nvidia':
            print("Installing ONNX Runtime with CUDA support...")
            onnx_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'onnxruntime-gpu'
            ]
        
        elif install_type == 'amd':
            print("Installing ONNX Runtime with ROCm support...")
            # Try ROCm version, fallback to CPU if not available
            onnx_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'onnxruntime-rocm==1.22.2.post1'
            ]
            if not run_command(onnx_cmd, "Installing ONNX Runtime (ROCm)", check=False):
                print("\n⚠️  ROCm version not available, falling back to CPU version")
                onnx_cmd = [sys.executable, '-m', 'pip', 'install', 'onnxruntime']
        
        else:  # apple or cpu
            print("Installing ONNX Runtime (CPU)...")
            onnx_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'onnxruntime'
            ]
        
        if not run_command(onnx_cmd, "Installing ONNX Runtime"):
            print("\n❌ Failed to install ONNX Runtime")
            print("\nTry manual installation:")
            print(f"  {' '.join(onnx_cmd)}")
            sys.exit(1)
        
        print("\n>>>> ONNX Runtime installed")
    else:
        print("\n----  Skipping ONNX Runtime installation (--skip-onnx)")
    
    # Verify installation
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    success = True

    # Test PyTorch
    try:
        import torch
        print(f"\n>>>> PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            if torch.version.hip:
                print(f">>>> AMD GPU: {torch.cuda.get_device_name(0)}")
                print(f"   ROCm version: {torch.version.hip}")
            else:
                print(f">>>> NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   VRAM: {memory:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f">>>> Apple Silicon MPS available")
        else:
            print(f"!!!!  CPU only (no GPU acceleration)")
    except ImportError:
        print(f"\nxxxx PyTorch import failed")
        success = False

    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"\n>>>> ONNX Runtime {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"   Providers: {', '.join(providers)}")

        if 'CUDAExecutionProvider' in providers:
            print(f"   >>>> CUDA acceleration available")
        elif 'ROCMExecutionProvider' in providers or 'MIGraphXExecutionProvider' in providers:
            print(f"   >>>> ROCm acceleration available")
        else:
            print(f"   !!!!  CPU only")
    except ImportError:
        print(f"\nxxxx ONNX Runtime import failed")
        success = False

    # Test Ultralytics
    try:
        from ultralytics import YOLO
        print(f"\n>>>> Ultralytics YOLO available")
    except ImportError:
        print(f"\nxxxx Ultralytics import failed")
        success = False

    # Summary
    print(f"\n{'='*60}")
    if success:
        print(">>>> INSTALLATION COMPLETE!")
    else:
        print("!!!!  INSTALLATION COMPLETED WITH WARNINGS")
    print(f"{'='*60}")

    print("\nNext steps:")
    print("  1. Validate environment: python pre_flight_check.py")
    print("  2. Download dataset:     python data_download.py")
    print("  3. Train model:          python train_baseline.py")
    print("  4. Analyze results:      python auto_analyze.py")
    print("\nOr run everything at once:")
    print("  python start.py")

    print(f"\n{'='*60}")
    print("==== WHAT TO RUN NEXT")
    print(f"{'='*60}")
    
    if success:
        print("\n>>>> Installation complete! Run these commands in order:")
        print("\n1.  Validate environment (recommended):")
        print("   python pre_flight_check.py")
        print("\n2.  Then choose ONE option:")
        print("\n   Option A - Complete Workflow (Easiest):")
        print("   python start.py")
        print("\n   Option B - Step by Step:")
        print("   python data_download.py       # Download dataset")
        print("   python train_baseline.py      # Train model (15-30 min)")
        print("   python auto_analyze.py        # Generate reports")
        print("\n   Option C - Quick Test First:")
        print("   python quick_test.py          # 1 min validation")
        print("   python start.py               # Then full workflow")
    else:
        print("\n!!!!  Installation had issues. Try:")
        print("   python install.py --force-cpu    # Force CPU")
        print("   pip cache purge                  # Clear cache")
        print("   python install.py                # Try again")
    
    print(f"\n{'='*60}\n")
    
    return success

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nxxxx Installation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)