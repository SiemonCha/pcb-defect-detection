"""
Comprehensive Project Validation
Tests all components and identifies issues before training.

Usage:
    python validate_project.py
"""

import os
import sys
import ast
import glob
from pathlib import Path

class Validator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def error(self, msg):
        self.errors.append(f"xxxx ERROR: {msg}")
        print(f"xxxx {msg}")
        
    def warn(self, msg):
        self.warnings.append(f"!!!!  WARNING: {msg}")
        print(f"!!!!  {msg}")
        
    def success(self, msg):
        self.passed.append(f">>>> {msg}")
        print(f">>>> {msg}")

validator = Validator()

print("="*60)
print("PROJECT VALIDATION")
print("="*60)

# 1. Check Python version
print("\n### Python Environment ###")
import platform
py_ver = platform.python_version()
major, minor = map(int, py_ver.split('.')[:2])
if major == 3 and minor == 11:
    validator.success(f"Python 3.11.x detected ({py_ver})")
else:
    validator.error(f"Python 3.11 required, found {py_ver}")

# 2. Check critical files exist
print("\n### File Structure ###")
critical_files = [
    'install.py',
    'data_download.py',
    'train_baseline.py',
    'train_production.py',
    'evaluate.py',
    'confusion_matrix.py',
    'export_onnx.py',
    'api.py',
    'transfer_learning.py',
    'auto_analyze.py',
    'requirements.txt',
    'tests/test_gpu.py',
    'tests/test_api.py',
]

for f in critical_files:
    if os.path.exists(f):
        validator.success(f"Found {f}")
    else:
        validator.error(f"Missing {f}")

# 3. Check for code issues
print("\n### Code Analysis ###")

# Check export_onnx.py for duplicate code
if os.path.exists('export_onnx.py'):
    with open('export_onnx.py', 'r') as f:
        content = f.read()
        # Count occurrences of the log saving block
        count = content.count('log_file = log_dir /')
        if count > 1:
            validator.error(f"export_onnx.py has duplicate code (log saving block repeated {count} times)")
        else:
            validator.success("export_onnx.py: No duplicate code detected")

# Check for syntax errors in all Python files
python_files = glob.glob('*.py') + glob.glob('scripts/*.py') + glob.glob('tests/*.py')
syntax_errors = []
for pf in python_files:
    try:
        with open(pf, 'r') as f:
            ast.parse(f.read())
    except SyntaxError as e:
        syntax_errors.append(f"{pf}: {e}")
        validator.error(f"Syntax error in {pf}: {e}")

if not syntax_errors:
    validator.success(f"All {len(python_files)} Python files have valid syntax")

# 4. Check requirements.txt issues
print("\n### Dependencies ###")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        reqs = f.read()
        if 'onnxruntime-rocm' in reqs and 'onnxruntime-gpu' not in reqs:
            validator.warn("requirements.txt contains AMD-specific onnxruntime-rocm")
            validator.warn("This will fail on NVIDIA/CPU systems")
            validator.warn("Solution: Remove onnxruntime-rocm from requirements.txt")
            validator.warn("Let install.py handle platform-specific installation")

# 5. Check dataset
print("\n### Dataset ###")
if os.path.exists('dataset_path.txt'):
    with open('dataset_path.txt', 'r') as f:
        dataset_path = f.read().strip()
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(data_yaml):
        validator.success(f"Dataset found: {data_yaml}")
    else:
        validator.error(f"data.yaml not found at {data_yaml}")
else:
    validator.warn("No dataset detected (run: python data_download.py)")

# 6. Check for trained models
print("\n### Trained Models ###")
model_patterns = [
    'runs/train/production_yolov8s*/weights/best.pt',
    'runs/train/baseline_yolov8n*/weights/best.pt',
]
found_models = []
for pattern in model_patterns:
    matches = glob.glob(pattern)
    if matches:
        found_models.extend(matches)

if found_models:
    validator.success(f"Found {len(found_models)} trained model(s)")
    for m in found_models:
        print(f"   → {m}")
else:
    validator.warn("No trained models (run: python train_baseline.py)")

# 7. Check imports
print("\n### Import Test ###")
try:
    import torch
    validator.success(f"PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        if torch.version.hip:
            validator.success(f"AMD GPU: {torch.cuda.get_device_name(0)}")
        else:
            validator.success(f"NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        validator.success("Apple Silicon (MPS)")
    else:
        validator.warn("CPU only (no GPU detected)")
except ImportError as e:
    validator.error(f"PyTorch import failed: {e}")

try:
    from ultralytics import YOLO
    validator.success("Ultralytics YOLO available")
except ImportError as e:
    validator.error(f"Ultralytics import failed: {e}")

try:
    import fastapi
    validator.success("FastAPI available")
except ImportError as e:
    validator.warn(f"FastAPI import failed (API won't work): {e}")

try:
    import matplotlib
    import seaborn
    validator.success("Visualization libraries available")
except ImportError as e:
    validator.warn(f"Visualization libraries missing: {e}")

# 8. Check workflow integration
print("\n### Workflow Integration ###")
workflow_file = 'auto_analyze.py'
if os.path.exists(workflow_file):
    with open(workflow_file, 'r') as f:
        content = f.read()
        if 'evaluate.py' in content and 'confusion_matrix.py' in content and 'export_onnx.py' in content:
            validator.success("auto_analyze.py integrates all analysis scripts")
        else:
            validator.warn("auto_analyze.py might be missing some integrations")

# Check if dataset validation scripts are documented
if os.path.exists('scripts/check_dataset.py'):
    validator.success("Dataset validation script exists")
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            if 'check_dataset.py' not in f.read():
                validator.warn("check_dataset.py not documented in README.md")
    else:
        validator.error("README.md missing")

# 9. Summary
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
print(f"\n>>>> Passed: {len(validator.passed)}")
print(f"!!!!  Warnings: {len(validator.warnings)}")
print(f"xxxx Errors: {len(validator.errors)}")

if validator.errors:
    print("\n### Critical Issues (MUST FIX) ###")
    for err in validator.errors:
        print(err)

if validator.warnings:
    print("\n### Warnings (SHOULD FIX) ###")
    for warn in validator.warnings:
        print(warn)

# 10. Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if not os.path.exists('dataset_path.txt'):
    print("\n1. Download dataset:")
    print("   python data_download.py")

if not found_models:
    print("\n2. Train baseline model:")
    print("   python train_baseline.py")

if 'onnxruntime-rocm' in open('requirements.txt').read():
    print("\n3. Fix requirements.txt:")
    print("   Remove 'onnxruntime-rocm==1.22.2.post1' line")
    print("   Let install.py handle platform-specific ONNX Runtime")

if os.path.exists('export_onnx.py'):
    with open('export_onnx.py', 'r') as f:
        if f.read().count('log_file = log_dir /') > 1:
            print("\n4. Fix export_onnx.py:")
            print("   Remove duplicate code at the end (lines 295-320)")

print("\n5. Run full workflow test:")
print("   python run_tests.py")

print("\n" + "="*60)
sys.exit(0 if len(validator.errors) == 0 else 1)