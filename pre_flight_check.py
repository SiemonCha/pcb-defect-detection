"""
Pre-Flight Check - Validate environment before training

Checks:
- Python version
- GPU availability
- Dependencies installed
- Dataset downloaded
- Disk space
- Memory available

Usage:
    python pre_flight_check.py
"""

import sys
import os
import platform
import shutil
import glob
from pathlib import Path

class CheckResult:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.errors = []
    
    def add_pass(self, msg):
        self.passed.append(msg)
        print(f">>>> {msg}")
    
    def add_warn(self, msg):
        self.warnings.append(msg)
        print(f"!!!!  {msg}")
    
    def add_error(self, msg):
        self.errors.append(msg)
        print(f"xxxx {msg}")
    
    def is_ready(self):
        return len(self.errors) == 0

result = CheckResult()

print("="*60)
print("PRE-FLIGHT CHECK")
print("="*60)

# 1. Python Version
print("\n### Python Environment ###")
py_ver = platform.python_version()
major, minor = map(int, py_ver.split('.')[:2])

if major == 3 and minor == 11:
    result.add_pass(f"Python 3.11.x ({py_ver})")
elif major == 3 and minor == 12:
    result.add_warn(f"Python 3.12 detected - 3.11 recommended ({py_ver})")
else:
    result.add_error(f"Python 3.11 required, found {py_ver}")

# 2. Check GPU
print("\n### GPU Detection ###")
try:
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if torch.version.hip:
            result.add_pass(f"AMD GPU: {device_name}")
            result.add_warn(f"ROCm detected - ensure onnxruntime-rocm installed")
        else:
            result.add_pass(f"NVIDIA GPU: {device_name} ({memory:.1f}GB)")
            if memory < 4:
                result.add_warn(f"Only {memory:.1f}GB VRAM - reduce batch size if training fails")
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        result.add_pass("Apple Silicon (MPS) detected")
        result.add_warn("MPS training may be slower than CUDA")
    
    else:
        result.add_warn("No GPU detected - training will be slow")
        result.add_warn("Consider using Google Colab or cloud GPU")

except ImportError:
    result.add_error("PyTorch not installed - run: python install.py")

# 3. Check Dependencies
print("\n### Dependencies ###")
required_packages = [
    ('torch', 'PyTorch'),
    ('ultralytics', 'YOLOv8'),
    ('roboflow', 'Dataset downloader'),
    ('fastapi', 'REST API'),
    ('matplotlib', 'Plotting'),
    ('seaborn', 'Visualization'),
]

for package, name in required_packages:
    try:
        __import__(package)
        result.add_pass(f"{name} installed")
    except ImportError:
        if package in ['fastapi']:
            result.add_warn(f"{name} missing - API won't work")
        else:
            result.add_error(f"{name} missing - run: python install.py")


# 4. Check Disk Space
print("\n### System Resources ###")
try:
    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024**3)
    
    if free_gb > 10:
        result.add_pass(f"Disk space: {free_gb:.1f}GB available")
    elif free_gb > 5:
        result.add_warn(f"Low disk space: {free_gb:.1f}GB (recommend >10GB)")
    else:
        result.add_error(f"Insufficient disk space: {free_gb:.1f}GB (need >5GB)")
except Exception as e:
    result.add_warn(f"Could not check disk space: {e}")

# 5. Check Memory
try:
    import psutil
    mem = psutil.virtual_memory()
    mem_gb = mem.total / (1024**3)
    
    if mem_gb > 8:
        result.add_pass(f"RAM: {mem_gb:.1f}GB")
    elif mem_gb > 4:
        result.add_warn(f"Limited RAM: {mem_gb:.1f}GB (recommend >8GB)")
    else:
        result.add_error(f"Insufficient RAM: {mem_gb:.1f}GB (need >4GB)")
except ImportError:
    result.add_warn("psutil not installed - cannot check memory")

# 6. Check for Previous Training
print("\n### Previous Training ###")
model_patterns = [
    'runs/train/baseline_yolov8n*/weights/best.pt',
    'runs/train/production_yolov8s*/weights/best.pt',
]

found_models = []
for pattern in model_patterns:
    matches = glob.glob(pattern)
    found_models.extend(matches)

if found_models:
    result.add_warn(f"Found {len(found_models)} existing trained model(s)")
    for m in found_models[:3]:  # Show first 3
        print(f"   → {m}")
    result.add_warn("Training will create new model in separate directory")

# 7. Check Write Permissions
print("\n### Permissions ###")
test_file = '.write_test'
try:
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    result.add_pass("Write permissions OK")
except Exception as e:
    result.add_error(f"Cannot write files: {e}")

# Create necessary directories
for dir_name in ['runs', 'logs', 'outputs']:
    try:
        Path(dir_name).mkdir(exist_ok=True)
        result.add_pass(f"Directory ready: {dir_name}/")
    except Exception as e:
        result.add_error(f"Cannot create {dir_name}/: {e}")

# Summary
print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n>>>> Passed:   {len(result.passed)}")
print(f"!!!!  Warnings: {len(result.warnings)}")
print(f"xxxx Errors:   {len(result.errors)}")

if result.errors:
    print("\n### Fix These First ###")
    for err in result.errors[:5]:  # Show first 5
        print(err)

if result.warnings:
    print("\n### Consider Fixing ###")
    for warn in result.warnings[:5]:  # Show first 5
        print(warn)

print("\n" + "="*60)

if result.is_ready():
    print(">>>> READY TO TRAIN")
    print("="*60)
    print("\n==== WHAT TO RUN NEXT - Choose ONE option:\n")
    print("Option A - Complete Workflow (Recommended):")
    print("  python start.py")
    print("  → Runs everything automatically (25-45 min)")
    print("  → Downloads dataset, trains, analyzes, generates reports")
    print("\nOption B - Step by Step:")
    print("  python data_download.py       # 1. Download dataset (2-5 min)")
    print("  python train_baseline.py      # 2. Train model (15-30 min)")
    print("  python auto_analyze.py        # 3. Generate reports (5 min)")
    print("\nOption C - Quick Test First:")
    print("  python quick_test.py          # 1. Quick validation (1 min)")
    print("  python start.py               # 2. Full workflow")
    print("\n---- Tip: Use 'python start.py' for easiest experience!")
    sys.exit(0)
else:
    print("xxxx NOT READY - Fix errors above")
    print("="*60)
    print("\n==== WHAT TO RUN NEXT:\n")
    
    if 'PyTorch not installed' in str(result.errors):
        print("1. Install dependencies:")
        print("   python install.py")
        print("\n2. Then run pre-flight check again:")
        print("   python pre_flight_check.py")
    else:
        print("Fix the errors shown above, then:")
        print("   python pre_flight_check.py")
    
    sys.exit(1)