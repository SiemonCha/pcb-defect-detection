"""
Hyperparameter Optimization using Optuna

Automatically finds best hyperparameters for training.

Usage:
    python -m training.hyperparameter                # Default 20 trials
    python -m training.hyperparameter --trials 50    # More thorough search
    python -m training.hyperparameter --quick        # Fast search (10 trials)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ultralytics import YOLO
import torch
import glob
import argparse
from pathlib import Path
import yaml
from data import resolve_dataset_yaml

def find_data_yaml() -> Path:
    return resolve_dataset_yaml()

def objective(trial, data_yaml: Path, device, epochs=30):
    """Optuna objective function"""
    
    # Hyperparameters to optimize
    lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True)
    lrf = trial.suggest_float('lrf', 0.001, 0.1, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.95)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 5)
    
    # Augmentation hyperparameters
    hsv_h = trial.suggest_float('hsv_h', 0.0, 0.05)
    hsv_s = trial.suggest_float('hsv_s', 0.3, 0.9)
    hsv_v = trial.suggest_float('hsv_v', 0.2, 0.6)
    degrees = trial.suggest_float('degrees', 0.0, 20.0)
    translate = trial.suggest_float('translate', 0.0, 0.2)
    scale = trial.suggest_float('scale', 0.3, 0.7)
    mosaic = trial.suggest_float('mosaic', 0.5, 1.0)
    mixup = trial.suggest_float('mixup', 0.0, 0.2)
    
    # Batch size
    batch = trial.suggest_categorical('batch', [4, 8, 16, 32])
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"lr0={lr0:.6f}, lrf={lrf:.4f}, momentum={momentum:.3f}")
    print(f"weight_decay={weight_decay:.6f}, batch={batch}")
    
    try:
        # Train model with suggested hyperparameters
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            project='runs/optuna',
            name=f'trial_{trial.number}',
            patience=5,
            save=False,  # Don't save intermediate models
            plots=False,
            verbose=False,
            # Optimizer params
            optimizer='AdamW',
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            cos_lr=True,
            # Augmentation params
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            flipud=0.5,
            fliplr=0.5,
            mosaic=mosaic,
            mixup=mixup,
            # Other
            cache='ram' if device != 'cpu' else False,
            amp=True if device == 'cuda' else False,
        )
        
        # Get validation mAP@0.5
        metrics = model.val(split='val', verbose=False)
        map50 = float(metrics.box.map50)
        
        print(f"Trial {trial.number} -> mAP@0.5: {map50:.4f}")
        
        return map50
        
    except Exception as e:
        print(f"xxxx Trial {trial.number} failed: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--quick', action='store_true', help='Quick search (10 trials, 20 epochs)')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        n_trials = 10
        epochs = 20
        print(f"\n>>>> QUICK MODE: {n_trials} trials, {epochs} epochs each")
    else:
        n_trials = args.trials
        epochs = 30
        print(f"\n>>>> FULL MODE: {n_trials} trials, {epochs} epochs each")
    
    # Setup
    data_yaml = find_data_yaml()
    
    # Get device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f">>>> Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f">>>> Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print(f">>>> Using CPU (WARNING: This will be slow)")
    
    print(f"{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dataset: {data_yaml}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {epochs}")
    print(f"Estimated time: {n_trials * epochs * 2 / 60:.1f} hours")
    
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print(f"\nxxxx Optuna not installed")
        print(f"   Install: pip install optuna")
        return
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    print(f"\n>>>> Starting optimization...")
    print(f"   Press Ctrl+C to stop early and save results\n")
    
    try:
        study.optimize(
            lambda trial: objective(trial, data_yaml, device, epochs),
            n_trials=n_trials,
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print(f"\n>>>> Optimization interrupted by user")
    
    # Results
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mAP@0.5: {study.best_value:.4f}")
    
    print(f"\nBest hyperparameters:")
    print("-"*60)
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Save results
    output_dir = Path('logs')
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best params as YAML
    best_params_path = output_dir / f'best_hyperparameters_{timestamp}.yaml'
    with open(best_params_path, 'w') as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    
    # Save report
    report_path = output_dir / f'hyperparameter_search_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("HYPERPARAMETER OPTIMIZATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {data_yaml}\n")
        f.write(f"Trials: {len(study.trials)}\n")
        f.write(f"Epochs per trial: {epochs}\n\n")
        
        f.write("Best Trial:\n")
        f.write("-"*60 + "\n")
        f.write(f"Trial number: {study.best_trial.number}\n")
        f.write(f"mAP@0.5: {study.best_value:.4f}\n\n")
        
        f.write("Best Hyperparameters:\n")
        f.write("-"*60 + "\n")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                f.write(f"{key:20s}: {value:.6f}\n")
            else:
                f.write(f"{key:20s}: {value}\n")
        
        f.write("\n\nTop 5 Trials:\n")
        f.write("-"*60 + "\n")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
        for i, trial in enumerate(sorted_trials[:5]):
            f.write(f"\n{i+1}. Trial {trial.number}: mAP@0.5 = {trial.value:.4f}\n")
            for key, value in trial.params.items():
                if isinstance(value, float):
                    f.write(f"   {key}: {value:.6f}\n")
                else:
                    f.write(f"   {key}: {value}\n")
        
        f.write("\n\nHow to use best hyperparameters:\n")
        f.write("-"*60 + "\n")
        f.write("Edit your training script with these values, or use:\n")
        f.write(f"  python {helper_path.name} --params logs/best_hyperparameters_*.yaml\n")
    
    # Visualization (if possible)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Trial values
        trials = [t.number for t in study.trials if t.value is not None]
        values = [t.value for t in study.trials if t.value is not None]
        
        ax1.plot(trials, values, 'o-', alpha=0.7)
        ax1.axhline(study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.4f}')
        ax1.set_xlabel('Trial', fontsize=12)
        ax1.set_ylabel('mAP@0.5', fontsize=12)
        ax1.set_title('Optimization History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Parameter importance (if enough trials)
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:10]  # Top 10
                importances = [importance[p] for p in params]
                
                ax2.barh(params, importances)
                ax2.set_xlabel('Importance', fontsize=12)
                ax2.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
            except:
                ax2.text(0.5, 0.5, 'Insufficient trials\nfor importance analysis', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'Insufficient trials\nfor importance analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        viz_path = output_dir / f'hyperparameter_optimization_{timestamp}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n>>>> Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"\n----- Could not generate visualization: {e}")
    
    print(f"\n>>>> Best parameters saved: {best_params_path}")
    print(f">>>> Report saved: {report_path}")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"1. Review best hyperparameters above")
    print(f"2. Train full model with optimized settings:")
    helper_path = output_dir / 'train_with_best_params.py'
    print(f"   python {helper_path.name} --params {best_params_path}")
    print("3. Or manually update your training configuration with the tuned values")
    
    # Create training script with best params
    if not helper_path.exists():
        helper_path.parent.mkdir(parents=True, exist_ok=True)
        with open(helper_path, 'w') as f:
            f.write(f'''"""
Train with optimized hyperparameters from Optuna search

Usage:
    python {helper_path.name} --params logs/best_hyperparameters_*.yaml
"""

import argparse
import yaml
from ultralytics import YOLO
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--model', type=str, default='yolov8s.pt')
parser.add_argument('--data', type=str, default=None)
args = parser.parse_args()

with open(args.params, 'r') as f:
    best_params = yaml.safe_load(f)

print("Loading model...", args.model)
model = YOLO(args.model)

train_kwargs = best_params.get('train_kwargs', {})
if args.data:
    train_kwargs['data'] = args.data

print("Training with best hyperparameters:")
for k, v in train_kwargs.items():
    print(f"  {k}: {v}")

model.train(**train_kwargs)
''')
        print(f"\n>>>> Helper script created: {helper_path}")

if __name__ == '__main__':
    main()
