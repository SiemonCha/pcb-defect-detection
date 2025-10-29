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
            logger.log(f"✅ {pkg} installed", 'SUCCESS')
        except ImportError:
            logger.log(f"❌ {pkg} missing", 'ERROR')
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
                logger.log(f"✅ AMD GPU detected: {device_name}", 'SUCCESS')
            else:
                memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.log(f"✅ NVIDIA GPU detected: {device_name} ({memory:.1f}GB)", 'SUCCESS')
        elif torch.backends.mps.is_available():
            logger.log(f"✅ Apple Silicon detected", 'SUCCESS')
        else:
            logger.log(f"⚠️  No GPU detected, using CPU", 'WARNING')
        
        return True
    except Exception as e:
        logger.log(f"❌ GPU check failed: {e}", 'ERROR')
        return False

def check_dataset():
    """Check if dataset exists"""
    logger.log("Checking dataset...")
    
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        
        if os.path.exists(data_yaml):
            logger.log(f"✅ Dataset found: {data_yaml}", 'SUCCESS')
            return True
    
    logger.log(f"❌ Dataset not found. Run: python data_download.py", 'ERROR')
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
            logger.log(f"✅ Trained model found: {model_path}", 'SUCCESS')
            return True
    
    logger.log(f"❌ No trained model found. Run: python train_baseline.py", 'ERROR')
    return False

def test_script(script_name, description, args=[], cwd=None):
    """Test a script by running it"""
    logger.log(f"\nTesting: {description}")
    logger.log(f"Running: python {script_name} {' '.join(args)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name] + args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd=cwd
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.log(f"✅ {description} completed in {elapsed:.1f}s", 'SUCCESS')
            if result.stdout:
                # Log last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        logger.log(f"   {line}", 'INFO')
            return True
        else:
            logger.log(f"❌ {description} failed: {result.stderr[:200]}", 'ERROR')
            return False
            
    except subprocess.TimeoutExpired:
        logger.log(f"❌ {description} timed out (>5min)", 'ERROR')
        return False
    except Exception as e:
        logger.log(f"❌ {description} error: {str(e)}", 'ERROR')
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
        logger.log("\n❌ Environment checks failed. Fix issues above.", 'ERROR')
        logger.save()
        return False
    
    # Phase 2: Core functionality tests (if model exists)
    if has_model:
        logger.log("\n### PHASE 2: Core Functionality Tests ###")
        
        # Test evaluation
        test_script('evaluate.py', 'Model Evaluation')
        
        # Test confusion matrix
        test_script('confusion_matrix.py', 'Confusion Matrix Analysis')
        
        # Test ONNX export
        test_script('export_onnx.py', 'ONNX Export')
        
        # Test GPU detection script (in tests folder)
        test_script('tests/test_gpu.py', 'GPU Detection')
    else:
        logger.log("\n⚠️  Skipping functionality tests (no trained model)", 'WARNING')
        logger.log("   Run 'python train_baseline.py' first", 'INFO')
    
    # Phase 3: File checks
    logger.log("\n### PHASE 3: Output File Checks ###")
    
    expected_files = [
        'export_onnx.py',
        'confusion_matrix.py',
        'api.py',
        'test_api.py',
        'transfer_learning.py',
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            logger.log(f"✅ {file} exists ({size:.1f} KB)", 'SUCCESS')
        else:
            logger.log(f"❌ {file} missing", 'ERROR')
    
    # Summary
    logger.log("\n### TEST SUMMARY ###")
    
    success_count = len([r for r in logger.results if '✅' in r])
    error_count = len([r for r in logger.results if '❌' in r])
    warning_count = len([r for r in logger.results if '⚠️' in r])
    
    logger.log(f"Passed: {success_count}")
    logger.log(f"Failed: {error_count}")
    logger.log(f"Warnings: {warning_count}")
    
    if error_count == 0:
        logger.log("\n🎉 All tests passed! Project ready for submission.", 'SUCCESS')
    elif has_model and error_count <= 2:
        logger.log("\n✅ Core functionality working. Minor issues detected.", 'SUCCESS')
    else:
        logger.log("\n⚠️  Some tests failed. Review errors above.", 'WARNING')
    
    logger.save()
    return error_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)