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

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

class WorkflowRunner:
    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []
    
    def run_step(self, script, description, args=[], required=True):
        """Run a workflow step"""
        print(f"\n{'='*60}")
        print(f"STEP: {description}")
        print(f"{'='*60}")
        print(f"Command: python {script} {' '.join(args)}")
        print()
        
        try:
            result = subprocess.run(
                [sys.executable, script] + args,
                capture_output=False,  # Show real-time output
                text=True
            )
            
            if result.returncode == 0:
                self.steps_completed.append(description)
                print(f"\n>>>> {description} completed")
                return True
            else:
                self.steps_failed.append(description)
                print(f"\nxxxx {description} failed")
                
                if required:
                    print(f"\nCritical step failed. Stopping workflow.")
                    return False
                return True
        
        except Exception as e:
            self.steps_failed.append(description)
            print(f"\nxxxx {description} error: {e}")
            
            if required:
                print(f"\nCritical step failed. Stopping workflow.")
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
            print("\n>>>> Completed Steps:")
            for step in self.steps_completed:
                print(f"   • {step}")
        
        if self.steps_failed:
            print("\nxxxx Failed Steps:")
            for step in self.steps_failed:
                print(f"   • {step}")

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
            if not runner.run_step('data_download.py', 'Dataset Download', required=True):
                runner.print_summary()
                sys.exit(1)
        else:
            print("\n✅ Dataset already downloaded (skip)")
            runner.steps_completed.append('Dataset Download (cached)')
    
    # Step 3: Validate dataset (if script exists)
    if os.path.exists('scripts/check_dataset.py'):
        runner.run_step('scripts/check_dataset.py', 'Dataset Validation', required=False)
    
    # Step 4: Train baseline model
    if not args.skip_train:
        # Check if model already exists
        import glob
        existing = glob.glob('runs/train/baseline_yolov8n*/weights/best.pt')
        
        if existing and not args.production:
            print(f"\n!!!!  Baseline model already exists: {existing[0]}")
            response = input("Train anyway? (y/N): ")
            if response.lower() != 'y':
                print("Skipping training (using existing model)")
                runner.steps_completed.append('Training (using existing)')
            else:
                if not runner.run_step('train_baseline.py', 'Baseline Training (15-30 min)', required=True):
                    runner.print_summary()
                    sys.exit(1)
        else:
            if not runner.run_step('train_baseline.py', 'Baseline Training (15-30 min)', required=True):
                runner.print_summary()
                sys.exit(1)
        
        # Step 5: Train production model (optional)
        if args.production:
            if not runner.run_step('train_production.py', 'Production Training (1-2 hours)', required=False):
                print("\n⚠️  Production training failed, but continuing with baseline model...")
    else:
        print("\n>>>> Training skipped (--skip-train)")
        runner.steps_completed.append('Training (skipped)')
    
    # Step 6: Run analysis
    if not runner.run_step('auto_analyze.py', 'Complete Analysis & Reporting', required=False):
        print("\n⚠️  Analysis failed, but models are still trained")
    
    # Step 7: Start API (optional)
    if args.api:
        print("\n" + "="*60)
        print("Starting API Server...")
        print("="*60)
        print("\nPress Ctrl+C to stop the server")
        print("API will be available at: http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        
        try:
            subprocess.run([sys.executable, 'api.py'])
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
        print(f"\n>>>> Generated {len(log_files)} log files in logs/")
        print("   • Evaluation reports")
        print("   • Confusion matrices")
        print("   • ONNX benchmarks")
    
    print("\n---- For Your Project Submission:")
    print("\n1. Collect Results:")
    print("   • All files from logs/ directory")
    print("   • Confusion matrix images")
    print("   • Model performance metrics")
    print("\n2. Review & Analyze:")
    print("   • Open confusion matrices to see per-class performance")
    print("   • Check ONNX speedup in benchmark files")
    print("   • Review evaluation metrics (mAP, precision, recall)")
    print("\n3. Write Report Including:")
    print("   • Model architecture (YOLOv8n baseline)")
    print("   • Training details (50 epochs, batch size, etc)")
    print("   • Performance results (from evaluation reports)")
    print("   • Confusion matrix analysis (strengths/weaknesses)")
    print("   • ONNX optimization results (speedup achieved)")
    print("   • Discussion of improvements and future work")
    
    if not args.api:
        print("\n---- Optional - Deploy API:")
        print("   Terminal 1: python api.py")
        print("   Terminal 2: python tests/test_api.py")
        print("   Browser: http://localhost:8000/docs (interactive API docs)")
    
    print("\n==== Optional - Train Production Model (Better Accuracy):")
    print("   python train_production.py    # YOLOv8s, 100 epochs, ~1-2 hours")
    print("   python auto_analyze.py        # Compare with baseline")
    
    print("\n==== All Your Results:")
    print(f"   • Logs: {Path('logs').absolute()}")
    print(f"   • Models: {Path('runs/train').absolute()}")
    print(f"   • Outputs: {Path('outputs').absolute() if Path('outputs').exists() else 'N/A'}")
    
    if args.production and 'Production Training' in runner.steps_completed:
        print("\n🎯 Model Comparison Available:")
        print("   • Baseline (YOLOv8n): Fast inference, good accuracy")
        print("   • Production (YOLOv8s): Better accuracy, slightly slower")
        print("   • Compare metrics in logs/ to choose best for your use case")
    
    print("\n" + "="*60)
    print(f"Workflow completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n>>>> Your PCB defect detection system is ready!")
    print("\n---- Quick Commands:")
    print("   View logs:    ls -lh logs/")
    print("   View models:  ls -lh runs/train/*/weights/")
    print("   Start API:    python api.py")
    print("   Re-analyze:   python auto_analyze.py")
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