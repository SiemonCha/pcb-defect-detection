"""
Auto-Run All Analysis and Logging

Analyzes ALL trained models (baseline + production) and generates logs.
Perfect for project submission - run once, get all results.

Usage:
    python auto_analyze.py
"""

import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.confusion import main as confusion_main
from evaluation.evaluate import main as evaluate_main
from deployment.export_onnx import main as export_onnx_main


def _run_callable(func, description: str, args: Optional[Iterable[str]] = None) -> bool:
    """Execute a module main() function with optional args."""
    print(f"\nRunning: {description}")
    try:
        if args is None:
            result = func()
        else:
            result = func(args)
    except SystemExit as exc:
        success = exc.code == 0
    except Exception as error:
        print(f"[ERROR] {description} error: {error}")
        return False
    else:
        if result is None or result is True or result == 0:
            success = True
        elif isinstance(result, int):
            success = result == 0
        else:
            success = bool(result)

    if success:
        print(f"[OK] {description} completed")
        return True

    print(f"[ERROR] {description} failed")
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
        print("\n[ERROR] No trained models found.")
        print("Please train at least one model first:")
        print("  python -m cli train-baseline      # Fast (15-30 min)")
        print("  python -m cli train-production    # Better accuracy (1-2 hours)")
        return False
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    print(f"\nAll logs will be saved to: {log_dir.absolute()}")
    print(f"\nFound {len(models_found)} model(s) to analyze:")
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
        results['evaluate'] = _run_callable(
            evaluate_main,
            f'{model_name} - Model Evaluation',
            args=[model_path],
        )
        
        # Confusion Matrix
        print("\n" + "-"*60)
        print("PHASE 2: Performance Analysis")
        print("-"*60)
        results['confusion_matrix'] = _run_callable(
            confusion_main,
            f'{model_name} - Confusion Matrix',
            args=[model_path],
        )
        
        # ONNX Export
        print("\n" + "-"*60)
        print("PHASE 3: Speed Optimization")
        print("-"*60)
        results['onnx_export'] = _run_callable(
            export_onnx_main,
            f'{model_name} - ONNX Export',
            args=[model_path],
        )
        
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
            status = "PASS" if success else "FAIL"
            print(f"  {status}: {task}")
    
    total_passed = sum(sum(r.values()) for r in all_results.values())
    total_tests = sum(len(r) for r in all_results.values())
    
    print(f"\nOverall: {total_passed}/{total_tests} completed successfully")
    
    if total_passed == total_tests:
        print("\nAll analysis completed successfully.")
    else:
        print("\n[WARN] Some analyses failed. Check errors above.")
    
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
                print("Evaluation reports:")
                for f in sorted(evaluations, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size:.1f} KB)")
            
            if confusion:
                print("\nPerformance analysis:")
                for f in sorted(confusion, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size:.1f} KB)")
            
            if onnx:
                print("\nONNX benchmarks:")
                for f in sorted(onnx, key=lambda x: x.stat().st_mtime, reverse=True):
                    size = f.stat().st_size / 1024
                    print(f"  - {f.name} ({size:.1f} KB)")
        else:
            print("No log files generated")
    
    # Model comparison
    if len(models_found) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print("\nBoth models are trained. Suggested comparisons:")
        print("\n  Baseline (YOLOv8n):")
        print("    - Faster inference")
        print("    - Smaller model size")
        print("    - Good for real-time applications")
        print("\n  Production (YOLOv8s):")
        print("    - Higher accuracy")
        print("    - Better for quality-critical applications")
        print("    - Recommended for final submission")
        print("\nReview logs/ to compare their performance metrics")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the logs/ directory for all reports")
    print("2. Compare confusion matrices for both models")
    print("3. Review ONNX speed benchmarks")
    print("4. Deploy API: python -m cli api")
    print("5. Test API: python tests/test_api.py")
    
    print("\nFor your project report, include:")
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