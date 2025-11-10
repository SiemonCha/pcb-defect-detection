"""
Comprehensive Testing Suite for PCB Defect Detection

Tests all components and logs results for project documentation.

Usage:
    python run_tests.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class TestLogger:
    def __init__(self, log_file='test_results.log'):
        self.log_file = log_file
        self.results = []
        
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        self.results.append(log_msg)
        
    def save(self):
        with open(self.log_file, 'w') as f:
            f.write('\n'.join(self.results))
        print(f"\n{'='*60}")
        print(f"Test log saved to: {self.log_file}")
        print(f"{'='*60}")

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.log("Checking dependencies...")
    
    required = [
        'torch',
        'ultralytics',
        'fastapi',
        'uvicorn',
        'matplotlib',
        'seaborn',
        'numpy',
        'PIL',
        'roboflow'
    ]
    
    missing = []
    for pkg in required:
        try:
            if pkg == 'PIL':
                __import__('PIL')
            else:
                __import__(pkg)
            logger.log(f"[OK] {pkg} installed", 'SUCCESS')
        except ImportError:
            logger.log(f"[MISSING] {pkg} not installed", 'ERROR')
            missing.append(pkg)
    
    if missing:
        logger.log(f"Missing packages: {', '.join(missing)}", 'ERROR')
        logger.log("Run: pip install matplotlib seaborn pillow", 'INFO')
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    logger.log("Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if torch.version.hip:
                logger.log(f"[OK] AMD GPU detected: {device_name}", 'SUCCESS')
            else:
                memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.log(f"[OK] NVIDIA GPU detected: {device_name} ({memory:.1f}GB)", 'SUCCESS')
        elif torch.backends.mps.is_available():
            logger.log("[OK] Apple Silicon detected", 'SUCCESS')
        else:
            logger.log("[WARN] No GPU detected, using CPU", 'WARNING')
        
        return True
    except Exception as e:
        logger.log(f"[ERROR] GPU check failed: {e}", 'ERROR')
        return False

def check_dataset():
    """Check if dataset exists"""
    logger.log("Checking dataset...")
    
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        
        if os.path.exists(data_yaml):
            logger.log(f"[OK] Dataset found: {data_yaml}", 'SUCCESS')
            return True
    
    logger.log("[ERROR] Dataset not found. Run: python -m cli data-download", 'ERROR')
    return False

def check_trained_model():
    """Check if model is trained"""
    logger.log("Checking trained models...")
    
    import glob
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            model_path = max(matches, key=os.path.getmtime)
            logger.log(f"[OK] Trained model found: {model_path}", 'SUCCESS')
            return True
    
    logger.log("[ERROR] No trained model found. Run: python -m cli train-baseline", 'ERROR')
    return False

def test_script(command, description, args=None, cwd=None, cli=False):
    """Test a command by running it"""
    logger.log(f"\nTesting: {description}")
    args = args or []
    if cli:
        display = f"python -m cli {command} {' '.join(args)}"
        cmd = [sys.executable, "-m", "cli", command] + args
    else:
        display = f"python {command} {' '.join(args)}"
        cmd = [sys.executable, command] + args

    logger.log(f"Running: {display}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd=cwd
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.log(f"[OK] {description} completed in {elapsed:.1f}s", 'SUCCESS')
            if result.stdout:
                # Log last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        logger.log(f"   {line}", 'INFO')
            return True
        else:
            logger.log(f"[ERROR] {description} failed: {result.stderr[:200]}", 'ERROR')
            return False
            
    except subprocess.TimeoutExpired:
        logger.log(f"[ERROR] {description} timed out (>5min)", 'ERROR')
        return False
    except Exception as e:
        logger.log(f"[ERROR] {description} error: {str(e)}", 'ERROR')
        return False

def main():
    global logger
    logger = TestLogger('test_results.log')
    
    logger.log("="*60)
    logger.log("PCB DEFECT DETECTION - TEST SUITE")
    logger.log("="*60)
    
    # Phase 1: Environment checks
    logger.log("\n### PHASE 1: Environment Checks ###")
    
    checks_passed = True
    checks_passed &= check_dependencies()
    checks_passed &= check_gpu()
    checks_passed &= check_dataset()
    
    has_model = check_trained_model()
    
    if not checks_passed:
        logger.log("\n[ERROR] Environment checks failed. Fix issues above.", 'ERROR')
        logger.save()
        return False
    
    # Phase 2: Core functionality tests (if model exists)
    if has_model:
        logger.log("\n### PHASE 2: Core Functionality Tests ###")
        
        # Test evaluation
        test_script('evaluate', 'Model Evaluation', cli=True)

        # Test confusion matrix
        test_script('confusion', 'Confusion Matrix Analysis', cli=True)
        
        # Test ONNX export
        test_script('export-onnx', 'ONNX Export', cli=True)
        
        # Test GPU detection script (in tests folder)
        test_script('tests/test_gpu.py', 'GPU Detection')
    else:
        logger.log("\n[WARN] Skipping functionality tests (no trained model)", 'WARNING')
        logger.log("   Run 'python -m cli train-baseline' first", 'INFO')
    
    # Phase 3: File checks
    logger.log("\n### PHASE 3: Output File Checks ###")
    
    expected_files = [
        'tests/test_api.py',
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            logger.log(f"[OK] {file} exists ({size:.1f} KB)", 'SUCCESS')
        else:
            logger.log(f"[ERROR] {file} missing", 'ERROR')
    
    # Summary
    logger.log("\n### TEST SUMMARY ###")
    
    success_count = len([r for r in logger.results if '[OK]' in r])
    error_count = len([r for r in logger.results if '[ERROR]' in r])
    warning_count = len([r for r in logger.results if '[WARN]' in r])
    
    logger.log(f"Passed: {success_count}")
    logger.log(f"Failed: {error_count}")
    logger.log(f"Warnings: {warning_count}")
    
    if error_count == 0:
        logger.log("\n[OK] All tests passed! Project ready for submission.", 'SUCCESS')
    elif has_model and error_count <= 2:
        logger.log("\n[OK] Core functionality working. Minor issues detected.", 'SUCCESS')
    else:
        logger.log("\n[WARN] Some tests failed. Review errors above.", 'WARNING')
    
    logger.save()
    return error_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)