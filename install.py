#!/usr/bin/env python3
import platform
import subprocess
import sys
import os

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

def install_requirements():
    # Get system info
    system = platform.system().lower()
    gpu_type = get_gpu_type()
    
    # Install common requirements first
    print("Installing common requirements...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/requirements-common.txt'])
    
    # Install platform-specific requirements
    if system == 'darwin' and platform.processor() == 'arm':
        print("Installing requirements for Mac (Apple Silicon)...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/requirements-mac.txt'])
    elif gpu_type == 'nvidia':
        print("Installing requirements for NVIDIA GPU...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/requirements-cuda.txt'])
    elif gpu_type == 'amd':
        print("Installing requirements for AMD GPU...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/requirements-rocm.txt'])
    else:
        print("No GPU detected or unsupported GPU type. Installing CPU-only version...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])

if __name__ == '__main__':
    install_requirements()