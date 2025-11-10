"""
One-Command Complete Workflow

Runs the entire PCB defect detection pipeline:
1. Environment validation
2. Dataset download (if needed)
3. Model training
4. Analysis & reporting

Usage:
    python start.py                  # Full workflow (baseline model)
    python start.py --production     # Train production model too
    python start.py --skip-train     # Only analysis (if model exists)
    python start.py --api            # Start API after training
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from glob import glob

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import data_download
from setup import check_dataset
from training import baseline as baseline_training
from training import production as production_training
import auto_analyze

class WorkflowRunner:
    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []
    
    def _run_callable(self, func, args):
        try:
            if args:
                result = func(args=args)
            else:
                result = func()
        except SystemExit as exc:
            success = exc.code == 0
        except Exception as error:
            print(f"\nxxxx {func.__name__} error: {error}")
            return False
        else:
            if result is None or result is True or result == 0:
                success = True
            elif isinstance(result, int):
                success = result == 0
            else:
                success = bool(result)
        if success:
            return True
        print(f"\nxxxx {func.__name__} failed")
        return False

    def run_step(self, target, description, args=None, required=True):
        """Run a workflow step"""
        print(f"\n{'='*60}")
        print(f"STEP: {description}")
        print(f"{'='*60}")

        if callable(target):
            print(f"Executing: {target.__module__}.{target.__name__}")
            success = self._run_callable(target, args or [])
        else:
            cmd_args = args or []
            print(f"Command: python {target} {' '.join(cmd_args)}\n")
            try:
                result = subprocess.run(
                    [sys.executable, target] + cmd_args,
                    capture_output=False,
                    text=True,
                )
                success = result.returncode == 0
            except Exception as error:
                print(f"\nxxxx {description} error: {error}")
                success = False

        if success:
            self.steps_completed.append(description)
            print(f"\n>>>> {description} completed")
            return True

        self.steps_failed.append(description)
        print(f"\nxxxx {description} failed")
        
        if required:
            print("\nCritical step failed. Stopping workflow.")
            return False
        return True
    
    def print_summary(self):
        """Print workflow summary"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        print(f"\nTotal time: {hours}h {minutes}m")
        print(f"Completed: {len(self.steps_completed)}")
        print(f"Failed: {len(self.steps_failed)}")
        
        if self.steps_completed:
            print("\nCompleted steps:")
            for step in self.steps_completed:
                print(f"   - {step}")

        if self.steps_failed:
            print("\nFailed steps:")
            for step in self.steps_failed:
                print(f"   - {step}")

def main():
    parser = argparse.ArgumentParser(description="Complete PCB Detection Workflow")
    parser.add_argument('--production', action='store_true', 
                       help='Train production model (YOLOv8s, 100 epochs)')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training (only run analysis)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download')
    parser.add_argument('--api', action='store_true',
                       help='Start API server after completion')
    
    args = parser.parse_args()
    
    runner = WorkflowRunner()
    
    print("\n" + "="*60)
    print("PCB DEFECT DETECTION - COMPLETE WORKFLOW")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.skip_train:
        print("Mode: Analysis only (skip training)")
    elif args.production:
        print("Mode: Full workflow + Production model")
    else:
        print("Mode: Full workflow (Baseline model)")
    
    # Step 1: Pre-flight check
    if not runner.run_step('pre_flight_check.py', 'Pre-Flight Check', required=False):
        print("\n!!!!  Pre-flight check failed, but continuing anyway...")
        print("Some features may not work. Fix errors when possible.")
    
    # Step 2: Download dataset (if needed)
    if not args.skip_download:
        if not os.path.exists('dataset_path.txt'):
            if not runner.run_step(data_download.main, 'Dataset Download', required=True):
                runner.print_summary()
                sys.exit(1)
        else:
            print("\nDataset already downloaded (skip)")
            runner.steps_completed.append('Dataset Download (cached)')
    
    # Step 3: Validate dataset
    runner.run_step(check_dataset.main, 'Dataset Validation', args=[], required=False)
    
    # Step 4: Train baseline model
    if not args.skip_train:
        # Check if model already exists
        existing = glob('runs/train/baseline_yolov8n*/weights/best.pt')
        
        if existing and not args.production:
            print(f"\n!!!!  Baseline model already exists: {existing[0]}")
            response = input("Train anyway? (y/N): ")
            if response.lower() != 'y':
                print("Skipping training (using existing model)")
                runner.steps_completed.append('Training (using existing)')
            else:
                if not runner.run_step(baseline_training.main, 'Baseline Training (15-30 min)', required=True):
                    runner.print_summary()
                    sys.exit(1)
        else:
            if not runner.run_step(baseline_training.main, 'Baseline Training (15-30 min)', required=True):
                runner.print_summary()
                sys.exit(1)
        
        # Step 5: Train production model (optional)
        if args.production:
            if not runner.run_step(production_training.main, 'Production Training (1-2 hours)', required=False):
                print("\nWARN: production training failed, continuing with baseline model")
    else:
        print("\nTraining skipped (--skip-train)")
        runner.steps_completed.append('Training (skipped)')
    
    # Step 6: Run analysis
    if not runner.run_step(auto_analyze.main, 'Complete Analysis & Reporting', required=False):
        print("\nWARN: analysis failed, but models remain available")
    
    # Step 7: Start API (optional)
    if args.api:
        print("\n" + "="*60)
        print("Starting API Server...")
        print("="*60)
        print("\nPress Ctrl+C to stop the server")
        print("API will be available at: http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        
        try:
            subprocess.run([sys.executable, '-m', 'cli', 'api'])
        except KeyboardInterrupt:
            print("\n\nAPI server stopped")
    
    # Print summary
    runner.print_summary()
    
    # Final instructions
    print("\n" + "="*60)
    print("==== WHAT TO DO NEXT")
    print("="*60)
    
    if os.path.exists('logs'):
        log_files = list(Path('logs').glob('*'))
        print(f"\nGenerated {len(log_files)} log files in logs/")
        print("   - Evaluation reports")
        print("   - Confusion matrices")
        print("   - ONNX benchmarks")
    
    print("\n---- For Your Project Submission:")
    print("\n1. Collect results:")
    print("   - All files from logs/ directory")
    print("   - Confusion matrix images")
    print("   - Model performance metrics")
    print("\n2. Review and analyse:")
    print("   - Inspect confusion matrices for per-class performance")
    print("   - Check ONNX speed benchmarks")
    print("   - Review evaluation metrics (mAP, precision, recall)")
    print("\n3. Report outline:")
    print("   - Model architecture (YOLOv8n baseline)")
    print("   - Training configuration (epochs, batch size, etc.)")
    print("   - Evaluation metrics summary")
    print("   - Confusion matrix discussion (strengths/weaknesses)")
    print("   - ONNX optimisation results")
    print("   - Recommendations for future work")
    
    if not args.api:
        print("\n---- Optional - Deploy API:")
        print("   Terminal 1: python -m cli api")
        print("   Terminal 2: python tests/test_api.py")
        print("   Browser: http://localhost:8000/docs")
    
    print("\n==== Optional - Train Production Model (Better Accuracy):")
    print("   python -m cli train-production    # YOLOv8s, 100 epochs, ~1-2 hours")
    print("   python auto_analyze.py                                   # Compare with baseline")
    
    print("\n==== All Your Results:")
    print(f"   - Logs: {Path('logs').absolute()}")
    print(f"   - Models: {Path('runs/train').absolute()}")
    print(f"   - Outputs: {Path('outputs').absolute() if Path('outputs').exists() else 'N/A'}")
    
    if args.production and 'Production Training' in runner.steps_completed:
        print("\nModel comparison available:")
        print("   - Baseline (YOLOv8n): fast inference, solid accuracy")
        print("   - Production (YOLOv8s): higher accuracy, slightly slower")
        print("   - Review logs to choose the appropriate model")
    
    print("\n" + "="*60)
    print(f"Workflow completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nSystem ready.")
    print("\nQuick commands:")
    print("   View logs:    ls -lh logs/")
    print("   View models:  ls -lh runs/train/*/weights/")
    print("   Start API:    python -m cli api")
    print("   Re-run analysis: python auto_analyze.py")
    print()
    
    sys.exit(0 if len(runner.steps_failed) == 0 else 1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nxxxx Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)