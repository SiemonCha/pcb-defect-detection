#!/usr/bin/env python3
"""
Quick Test - Rapid validation of the complete system

This script tests the most critical components quickly.
For full testing, see TESTING_GUIDE.md

Usage:
    python quick_test.py
"""

import subprocess
import sys
import os
import time

def run_cmd(cmd, description):
    """Run command and report status"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f">>>> PASS ({elapsed:.1f}s)")
            return True
        else:
            print(f"xxxx FAIL ({elapsed:.1f}s)")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("QUICK TEST - System Validation")
    print("="*60)
    print("This runs critical tests. For full testing, see TESTING_GUIDE.md")
    
    results = {}
    
    # Test 1: Python version
    print("\n### TEST 1: Python Version ###")
    import platform
    py_ver = platform.python_version()
    major, minor = map(int, py_ver.split('.')[:2])
    
    if major == 3 and minor >= 11:
        print(f">>>> PASS: Python {py_ver}")
        results['python'] = True
    else:
        print(f"xxxx FAIL: Python {py_ver} (need 3.11+)")
        results['python'] = False
    
    # Test 2: Imports
    print("\n### TEST 2: Critical Imports ###")
    try:
        import torch
        print(f">>>> PASS: PyTorch {torch.__version__}")
        results['torch'] = True
    except ImportError:
        print(f"xxxx FAIL: PyTorch not installed")
        results['torch'] = False
    
    try:
        from ultralytics import YOLO
        print(f">>>> PASS: Ultralytics YOLO")
        results['yolo'] = True
    except ImportError:
        print(f"xxxx FAIL: Ultralytics not installed")
        results['yolo'] = False
    
    # Test 3: GPU Detection
    print("\n### TEST 3: GPU Detection ###")
    try:
        import torch
        if torch.cuda.is_available():
            print(f">>>> PASS: GPU detected - {torch.cuda.get_device_name(0)}")
            results['gpu'] = True
        else:
            print(f"!!!!  WARNING: No GPU (CPU only)")
            results['gpu'] = False
    except:
        print(f"!!!!  WARNING: Cannot check GPU")
        results['gpu'] = False
    
    # Test 4: Dataset
    print("\n### TEST 4: Dataset ###")
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        
        if os.path.exists(data_yaml):
            print(f">>>> PASS: Dataset found at {data_yaml}")
            results['dataset'] = True
        else:
            print(f"xxxx FAIL: data.yaml not found")
            results['dataset'] = False
    else:
        print(f"❌ FAIL: Dataset not downloaded (run: python data_download.py)")
        results['dataset'] = False
    
    # Test 5: File Structure
    print("\n### TEST 5: Required Files ###")
    required_files = [
        'train_baseline.py',
        'evaluate.py',
        'confusion_matrix.py',
        'export_onnx.py',
        'api.py',
        'auto_analyze.py'
    ]
    
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f">>>> {f}")
        else:
            print(f"❌ {f} missing")
            missing.append(f)
    
    results['files'] = len(missing) == 0
    
    # Test 6: Trained Model (optional)
    print("\n### TEST 6: Trained Model ###")
    import glob
    models = glob.glob('runs/train/*/weights/best.pt')
    
    if models:
        print(f">>>> PASS: Found {len(models)} trained model(s)")
        for m in models[:3]:
            print(f"   → {m}")
        results['model'] = True
    else:
        print(f"!!!!  WARNING: No trained model (run: python train_baseline.py)")
        results['model'] = False
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    critical = sum([results.get('python', False), 
                   results.get('torch', False), 
                   results.get('yolo', False),
                   results.get('files', False)])
    
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Critical: {critical}/4")
    
    print("\nResults:")
    for test, status in results.items():
        symbol = ">>>>" if status else ("!!!!" if test in ['gpu', 'model', 'dataset'] else "xxxx")
        print(f"  {symbol} {test}")
    
    print("\n" + "="*60)
    
    if critical == 4:
        if passed == total:
            print(">>>> ALL TESTS PASSED")
            print("="*60)
            print("\nYour system is fully ready!")
            print("\n==== WHAT TO RUN NEXT - Choose ONE:\n")
            print("Option A - Complete Workflow (Easiest):")
            print("  python start.py")
            print("  -> Automates everything: dataset, training, analysis")
            print("  -> Takes 25-45 min (GPU) or 2-3 hours (CPU)")
            print("\nOption B - Detailed Validation First:")
            print("  python pre_flight_check.py    # Full environment check")
            print("  python start.py               # Then full workflow")
            print("\nOption C - Manual Step by Step:")
            print("  python data_download.py       # 1. Get dataset")
            print("  python train_baseline.py      # 2. Train (15-30 min)")
            print("  python auto_analyze.py        # 3. Analyze & report")
            print("\n---- Recommended: python start.py")
        else:
            print("!!!!  MOSTLY READY")
            print("="*60)
            print("\nCore system working, but some optional features missing:")
            if not results.get('dataset'):
                print("  • Dataset: python data_download.py")
            if not results.get('model'):
                print("  • Model: python train_baseline.py")
            if not results.get('gpu'):
                print("  • GPU not detected (CPU training will be slower)")
            print("\n==== WHAT TO RUN NEXT:\n")
            print("1. Fix warnings above (optional)")
            print("2. Then run:")
            print("   python start.py")
    else:
        print("xxxx SETUP INCOMPLETE")
        print("="*60)
        print("\nCritical issues found. Fix these first:")
        if not results.get('python'):
            print("  • Install Python 3.11 or 3.12")
        if not results.get('torch'):
            print("  • Run: python install.py")
        if not results.get('yolo'):
            print("  • Run: pip install -r requirements.txt")
        if not results.get('files'):
            print("  • Ensure all project files are present")
        print("\n📋 WHAT TO RUN NEXT:\n")
        print("1. Fix critical errors above")
        print("2. Run quick_test.py again:")
        print("   python quick_test.py")
        print("3. When all pass, run:")
        print("   python start.py")
    
    print("\n📚 For detailed help:")
    print("  • Full validation: python pre_flight_check.py")
    print("  • Testing guide: See TESTING_GUIDE.md")
    print("  • Setup help: See HOW_TO_TEST.md")
    print()
    
    return critical == 4

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)