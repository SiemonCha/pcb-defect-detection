"""
K-Fold Cross-Validation - Get confidence intervals for model performance

Splits data into K folds and trains K models to estimate true performance.

Usage:
    python -m training.cross_validation                  # 5-fold CV
    python -m training.cross_validation --folds 10       # 10-fold CV
    python -m training.cross_validation --quick          # 3-fold with fewer epochs
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
import shutil
import random
from collections import defaultdict
import numpy as np

def find_data_yaml():
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml):
            return data_yaml
    patterns = ['data/*/data.yaml', 'data/data.yaml']
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError("data.yaml not found")

def create_fold_splits(dataset_root, n_folds=5, seed=42):
    """Create K-fold splits from train+val data"""
    random.seed(seed)
    
    # Collect all training images
    train_img_dir = os.path.join(dataset_root, 'train', 'images')
    valid_img_dir = os.path.join(dataset_root, 'valid', 'images')
    
    all_images = []
    
    if os.path.exists(train_img_dir):
        all_images.extend(glob.glob(os.path.join(train_img_dir, '*.jpg')))
        all_images.extend(glob.glob(os.path.join(train_img_dir, '*.png')))
    
    if os.path.exists(valid_img_dir):
        all_images.extend(glob.glob(os.path.join(valid_img_dir, '*.jpg')))
        all_images.extend(glob.glob(os.path.join(valid_img_dir, '*.png')))
    
    if not all_images:
        raise ValueError("No training/validation images found")
    
    # Shuffle
    random.shuffle(all_images)
    
    # Split into folds
    fold_size = len(all_images) // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(all_images)
        folds.append(all_images[start_idx:end_idx])
    
    return folds

def create_fold_dataset(dataset_root, folds, val_fold_idx, cv_root):
    """Create dataset for specific fold"""
    fold_dir = cv_root / f'fold_{val_fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train and val directories
    train_img_dir = fold_dir / 'train' / 'images'
    train_lbl_dir = fold_dir / 'train' / 'labels'
    val_img_dir = fold_dir / 'val' / 'images'
    val_lbl_dir = fold_dir / 'val' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for fold_idx, fold_images in enumerate(folds):
        is_val = (fold_idx == val_fold_idx)
        
        for img_path in fold_images:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            # Find label file
            img_dir = os.path.dirname(img_path)
            label_dir = img_dir.replace('images', 'labels')
            label_path = os.path.join(label_dir, label_name)
            
            # Copy to appropriate split
            if is_val:
                shutil.copy(img_path, val_img_dir / img_name)
                if os.path.exists(label_path):
                    shutil.copy(label_path, val_lbl_dir / label_name)
            else:
                shutil.copy(img_path, train_img_dir / img_name)
                if os.path.exists(label_path):
                    shutil.copy(label_path, train_lbl_dir / label_name)
    
    # Create data.yaml for this fold
    with open(dataset_root / 'data.yaml', 'r') as f:
        original_config = yaml.safe_load(f)
    
    fold_config = {
        'path': str(fold_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': original_config['nc'],
        'names': original_config['names']
    }
    
    fold_yaml_path = fold_dir / 'data.yaml'
    with open(fold_yaml_path, 'w') as f:
        yaml.dump(fold_config, f, default_flow_style=False)
    
    return fold_yaml_path

def train_fold(fold_idx, data_yaml, device, epochs=50, project='runs/cv'):
    """Train one fold"""
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx + 1}")
    print(f"{'='*60}")
    
    try:
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=8,
            device=device,
            project=project,
            name=f'fold_{fold_idx}',
            patience=10,
            save=True,
            plots=False,
            verbose=False,
            cache='ram' if device != 'cpu' else False,
            amp=True if device == 'cuda' else False,
        )
        
        # Get validation metrics
        metrics = model.val(verbose=False)
        
        return {
            'fold': fold_idx,
            'map50': float(metrics.box.map50),
            'map': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'model_path': f"{project}/fold_{fold_idx}/weights/best.pt"
        }
        
    except Exception as e:
        print(f"xxxx Fold {fold_idx} training failed: {e}")
        return {
            'fold': fold_idx,
            'map50': 0.0,
            'map': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per fold')
    parser.add_argument('--quick', action='store_true', help='Quick mode (3 folds, 30 epochs)')
    args = parser.parse_args()
    
    if args.quick:
        n_folds = 3
        epochs = 30
        print(f"\n>>>> QUICK MODE: {n_folds} folds, {epochs} epochs")
    else:
        n_folds = args.folds
        epochs = args.epochs
        print(f"\n>>>> FULL MODE: {n_folds} folds, {epochs} epochs")
    
    # Setup
    data_yaml = find_data_yaml()
    dataset_root = Path(os.path.dirname(data_yaml))
    
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
    print("K-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Dataset: {data_yaml}")
    print(f"Folds: {n_folds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Estimated time: {n_folds * epochs * 2 / 60:.1f} hours")
    
    # Create fold splits
    print(f"\n>>>> Creating {n_folds}-fold splits...")
    folds = create_fold_splits(dataset_root, n_folds)
    
    print(f">>>> Fold sizes:")
    for i, fold in enumerate(folds):
        print(f"   Fold {i+1}: {len(fold)} images")
    
    # Create CV directory
    cv_root = Path('runs/cv')
    cv_root.mkdir(parents=True, exist_ok=True)
    
    # Train each fold
    fold_results = []
    
    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Create dataset for this fold
        fold_yaml = create_fold_dataset(dataset_root, folds, fold_idx, cv_root)
        print(f">>>> Created fold dataset: {fold_yaml}")
        
        # Train
        result = train_fold(fold_idx, fold_yaml, device, epochs)
        fold_results.append(result)
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"   mAP@0.5: {result['map50']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    successful_folds = [r for r in fold_results if 'error' not in r]
    
    if not successful_folds:
        print(f"xxxx All folds failed")
        return
    
    metrics = ['map50', 'map', 'precision', 'recall']
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in successful_folds]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"\nMetric                Mean +/- Std       [Min, Max]")
    print("-" * 60)
    for metric in metrics:
        s = stats[metric]
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:20s}  {s['mean']:.4f} +/- {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Calculate confidence interval (95%)
    import scipy.stats as st
    
    if len(successful_folds) >= 3:
        print(f"\n95% Confidence Intervals:")
        print("-" * 60)
        for metric in metrics:
            values = [r[metric] for r in successful_folds]
            ci = st.t.interval(0.95, len(values)-1, 
                              loc=np.mean(values), 
                              scale=st.sem(values))
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:20s}  [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Save results
    output_dir = Path('logs')
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_path = output_dir / f'cross_validation_{n_folds}fold_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("K-FOLD CROSS-VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {data_yaml}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Epochs per fold: {epochs}\n")
        f.write(f"Successful folds: {len(successful_folds)}/{n_folds}\n\n")
        
        f.write("Per-Fold Results:\n")
        f.write("-"*60 + "\n")
        for r in fold_results:
            if 'error' in r:
                f.write(f"Fold {r['fold']+1}: FAILED - {r['error']}\n")
            else:
                f.write(f"Fold {r['fold']+1}: mAP@0.5={r['map50']:.4f}, "
                       f"Precision={r['precision']:.4f}, Recall={r['recall']:.4f}\n")
        
        f.write("\n\nAggregated Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Metric':<20s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}\n")
        f.write("-"*60 + "\n")
        for metric in metrics:
            s = stats[metric]
            metric_name = metric.replace('_', ' ').title()
            f.write(f"{metric_name:<20s}  {s['mean']:>8.4f}  {s['std']:>8.4f}  "
                   f"{s['min']:>8.4f}  {s['max']:>8.4f}\n")
        
        if len(successful_folds) >= 3:
            f.write("\n\n95% Confidence Intervals:\n")
            f.write("-"*60 + "\n")
            for metric in metrics:
                values = [r[metric] for r in successful_folds]
                ci = st.t.interval(0.95, len(values)-1,
                                  loc=np.mean(values),
                                  scale=st.sem(values))
                metric_name = metric.replace('_', ' ').title()
                f.write(f"{metric_name:<20s}  [{ci[0]:.4f}, {ci[1]:.4f}]\n")
        
        f.write("\n\nInterpretation:\n")
        f.write("-"*60 + "\n")
        
        map50_mean = stats['map50']['mean']
        map50_std = stats['map50']['std']
        
        if map50_std < 0.02:
            f.write("Very consistent performance across folds (std < 0.02)\n")
            f.write("   Model is robust and reliable\n\n")
        elif map50_std < 0.05:
            f.write("Good consistency across folds (std < 0.05)\n")
            f.write("   Model performance is stable\n\n")
        else:
            f.write("High variance across folds (std >= 0.05)\n")
            f.write("   Performance depends on train/val split\n")
            f.write("   Consider:\n")
            f.write("   - Collecting more data\n")
            f.write("   - Increasing training epochs\n")
            f.write("   - Addressing class imbalance\n\n")
        
        if map50_mean > 0.85:
            f.write("Excellent average performance (mAP@0.5 > 85%)\n\n")
        elif map50_mean > 0.70:
            f.write("Good average performance (mAP@0.5 > 70%)\n\n")
        else:
            f.write("Performance below target (mAP@0.5 < 70%)\n")
            f.write("   Model needs improvement\n\n")
    
    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            values = [r[metric] for r in successful_folds]
            folds_nums = [r['fold']+1 for r in successful_folds]
            
            ax.plot(folds_nums, values, 'o-', linewidth=2, markersize=8)
            ax.axhline(stats[metric]['mean'], color='r', linestyle='--', 
                      label=f"Mean: {stats[metric]['mean']:.4f}")
            ax.fill_between(range(1, n_folds+1), 
                           stats[metric]['mean'] - stats[metric]['std'],
                           stats[metric]['mean'] + stats[metric]['std'],
                           alpha=0.2, color='r', label='+/- 1 Std')
            
            ax.set_xlabel('Fold', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()} Across Folds", 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xticks(range(1, n_folds+1))
        
        plt.tight_layout()
        viz_path = output_dir / f'cross_validation_{n_folds}fold_{timestamp}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n>>>> Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"\n----- Could not generate visualization: {e}")
    
    print(f"\n>>>> Report saved: {report_path}")
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"mAP@0.5: {stats['map50']['mean']:.4f} +/- {stats['map50']['std']:.4f}")
    print(f"\nYou can report model performance as:")
    print(f"  mAP@0.5 = {stats['map50']['mean']:.2%} +/- {stats['map50']['std']:.2%}")
    print(f"  ({n_folds}-fold cross-validation)")

if __name__ == '__main__':
    main()
