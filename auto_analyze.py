"""
Auto-Run All Analysis and Logging

Analyzes ALL trained models (baseline + production) and generates logs.
Perfect for project submission - run once, get all results.

Usage:
    python auto_analyze.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import glob

def run_script(script_name, description, args=[]):
    """Run a script and report status"""
    print(f"\nRunning: {description}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name] + args,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("AUTO-ANALYZE: Complete Analysis & Logging")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if models exist
    patterns = {
        'production': 'runs/train/production_yolov8s*/weights/best.pt',
        'baseline': 'runs/train/baseline_yolov8n*/weights/best.pt',
    }
    
    models_found = {}
    for name, pattern in patterns.items():
        matches = glob.glob(pattern)
        if matches:
            models_found[name] = max(matches, key=os.path.getmtime)
    
    if not models_found:
        print("\n❌ No trained models found.")
        print("Please train at least one model first:")
        print("  python train_baseline.py      # Fast (15-30 min)")
        print("  python train_production.py    # Better accuracy (1-2 hours)")
        return False
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 All logs will be saved to: {log_dir.absolute()}")
    print(f"\n🔍 Found {len(models_found)} model(s) to analyze:")
    for name, path in models_found.items():
        print(f"   - {name}: {path}")
    
    # Analyze each model found
    all_results = {}
    
    for model_name, model_path in models_found.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {model_name.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        
        results = {}
        
        # Evaluation
        print("\n" + "-"*60)
        print("PHASE 1: Evaluation")
        print("-"*60)
        results['evaluate'] = run_script('evaluate.py', f'{model_name} - Model Evaluation', [model_path])
        
        # Confusion Matrix
        print("\n" + "-"*60)
        print("PHASE 2: Performance Analysis")
        print("-"*60)
        results['confusion_matrix'] = run_script('confusion_matrix.py', f'{model_name} - Confusion Matrix', [model_path])
        
        # ONNX Export
        print("\n" + "-"*60)
        print("PHASE 3: Speed Optimization")
        print("-"*60)
        results['onnx_export'] = run_script('export_onnx.py', f'{model_name} - ONNX Export', [model_path])
        
        all_results[model_name] = results
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for model_name, results in all_results.items():
        passed = sum(results.values())
        total = len(results)
        print(f"\n{model_name.upper()} Model: {passed}/{total} completed")
        for task, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status}: {task}")
    
    total_passed = sum(sum(r.values()) for r in all_results.values())
    total_tests = sum(len(r) for r in all_results.values())
    
    print(f"\nOverall: {total_passed}/{total_tests} completed successfully")
    
    if total_passed == total_tests:
        print("\n🎉 All analysis completed successfully!")
    else:
        print("\n⚠️  Some analyses failed. Check errors above.")
    
    # Show generated files
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    if log_dir.exists():
        log_files = list(log_dir.glob('*'))
        if log_files:
            print(f"\nLogs directory: {log_dir.absolute()}")
            print(f"Total files: {len(log_files)}\n")
            
            # Group by type
            evaluations = [f for f in log_files if 'evaluation' in f.name]
            confusion = [f for f in log_files if 'confusion' in f.name or 'performance' in f.name]
            onnx = [f for f in log_files if 'onnx' in f.name or 'benchmark' in f.name]
            
            if evaluations:
                print("Evaluation Reports:")
                for f in sorted(evaluations, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  📄 {f.name} ({size:.1f} KB)")
            
            if confusion:
                print("\nPerformance Analysis:")
                for f in sorted(confusion, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  📊 {f.name} ({size:.1f} KB)")
            
            if onnx:
                print("\nONNX Benchmarks:")
                for f in sorted(onnx, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  ⚡ {f.name} ({size:.1f} KB)")
        else:
            print("No log files generated")
    
    # Model comparison
    if len(models_found) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print("\n✅ You have BOTH models trained! Great for comparison:")
        print("\n  Baseline (YOLOv8n):")
        print("    - Faster inference")
        print("    - Smaller model size")
        print("    - Good for real-time applications")
        print("\n  Production (YOLOv8s):")
        print("    - Higher accuracy")
        print("    - Better for quality-critical applications")
        print("    - Recommended for final submission")
        print("\n📊 Check logs/ to compare their performance metrics")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. ✅ Check logs/ directory for all reports")
    print("2. ✅ Review confusion matrices for both models")
    print("3. ✅ Compare ONNX speedup improvements")
    print("4. 🚀 Deploy API: python api.py")
    print("5. 🧪 Test API: python tests/test_api.py")
    
    print("\n📝 For your project report, include:")
    print("  - All files from logs/ directory")
    print("  - Comparison of baseline vs production (if both available)")
    print("  - Confusion matrix images showing per-class performance")
    print("  - ONNX benchmark showing speedup improvements")
    print("  - Discussion of model trade-offs (speed vs accuracy)")
    
    print(f"\n{'='*60}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    return total_passed == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)