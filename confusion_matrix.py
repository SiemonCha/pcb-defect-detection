"""
Generate confusion matrix and detailed per-class analysis

Usage:
    python confusion_matrix.py                   # Auto-detect best model
    python confusion_matrix.py runs/.../best.pt  # Specific model
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import glob
from pathlib import Path

def find_best_model():
    """Find the best trained model"""
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    
    raise FileNotFoundError("No trained model found. Train first.")

def find_data_yaml():
    """Find data.yaml"""
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
    
    raise FileNotFoundError("data.yaml not found. Run: python data_download.py")

def plot_confusion_matrix(matrix, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    
    # Normalize by row (true labels)
    matrix_norm = matrix.astype('float') / (matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    sns.heatmap(
        matrix_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names + ['Background'],
        yticklabels=class_names + ['Background'],
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>>>> Confusion matrix saved to: {save_path}")

def analyze_per_class(matrix, class_names):
    """Analyze per-class performance"""
    results = []
    
    # Only analyze classes that actually exist in the confusion matrix
    num_classes = min(len(class_names), matrix.shape[0] - 1)  # Exclude background
    
    for idx in range(num_classes):
        class_name = class_names[idx]
        
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        tn = matrix.sum() - tp - fp - fn
        
        # Skip if no support (no samples for this class)
        support = int(tp + fn)
        if support == 0:
            continue
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        results.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'support': support
        })
    
    return results

def identify_weaknesses(results):
    """Identify model weaknesses"""
    issues = []
    
    for r in results:
        # Low recall = missing detections
        if r['recall'] < 0.7:
            issues.append(f"❌ {r['class']}: Low recall ({r['recall']:.1%}) - missing {r['fn']} defects")
        
        # Low precision = false alarms
        if r['precision'] < 0.7:
            issues.append(f"⚠️  {r['class']}: Low precision ({r['precision']:.1%}) - {r['fp']} false alarms")
        
        # Class imbalance
        if r['support'] < 10:
            issues.append(f"📊 {r['class']}: Low support ({r['support']} samples) - needs more data")
    
    return issues

# Get paths
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = find_best_model()

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

data_yaml = find_data_yaml()

# Load model
print(f"==== Loading model: {model_path}")
model = YOLO(model_path)

# Run validation to get metrics
print(f"==== Running validation on test set...")
metrics = model.val(data=data_yaml, split='test', plots=False)

# Get confusion matrix
if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
    cm = metrics.confusion_matrix.matrix
    class_names = list(metrics.names.values())
    
    # Create output directories
    output_dir = Path(model_path).parent.parent / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Use metrics from YOLO validation directly (more reliable)
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Get per-class metrics from YOLO
    if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap'):
        print(f"\n{'Class':<25} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'Samples':>10}")
        print("-" * 75)
        
        # Get metrics arrays
        class_indices = metrics.box.ap_class_index
        ap50_values = metrics.box.ap50  # AP at IoU=0.5
        
        results_for_report = []
        
        for idx, class_idx in enumerate(class_indices):
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                ap50 = float(ap50_values[idx]) if idx < len(ap50_values) else 0.0
                
                # These are overall metrics - per class breakdown not directly available
                # So we show what YOLO provides
                print(f"{class_name:<25} {ap50:>10.1%} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                
                results_for_report.append({
                    'class': class_name,
                    'ap50': ap50
                })
        
        # Identify weaknesses based on mAP
        print("\n" + "="*60)
        print("IDENTIFIED WEAKNESSES")
        print("="*60)
        
        weak_classes = [r for r in results_for_report if r['ap50'] < 0.3]
        medium_classes = [r for r in results_for_report if 0.3 <= r['ap50'] < 0.6]
        strong_classes = [r for r in results_for_report if r['ap50'] >= 0.6]
        
        if weak_classes:
            print("\n❌ Poor Performance (mAP@0.5 < 30%):")
            for r in weak_classes:
                print(f"   {r['class']}: {r['ap50']:.1%}")
        
        if medium_classes:
            print("\n⚠️  Moderate Performance (mAP@0.5 30-60%):")
            for r in medium_classes:
                print(f"   {r['class']}: {r['ap50']:.1%}")
        
        if strong_classes:
            print("\n✅ Strong Performance (mAP@0.5 > 60%):")
            for r in strong_classes:
                print(f"   {r['class']}: {r['ap50']:.1%}")
        
        if not weak_classes and not medium_classes:
            print("\n✅ All classes performing well!")
    else:
        print("\nDetailed per-class metrics not available in this YOLO version.")
    
    # Save detailed report
    report_path = output_dir / 'performance_report.txt'
    with open(report_path, 'w') as f:
        f.write("DETAILED PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Overall mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"Overall Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Overall Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("Per-Class mAP@0.5:\n")
        f.write("-" * 60 + "\n")
        
        if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
            for idx, class_idx in enumerate(metrics.box.ap_class_index):
                if class_idx < len(class_names):
                    class_name = class_names[class_idx]
                    ap50 = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.0
                    f.write(f"{class_name:<30} {ap50:.4f}\n")
        
        f.write("\n\nRecommendations:\n")
        f.write("-" * 60 + "\n")
        
        if weak_classes:
            f.write(f"\n1. Improve detection for weak classes:\n")
            for r in weak_classes:
                f.write(f"   - {r['class']}\n")
            f.write("   → Collect more training samples\n")
            f.write("   → Increase data augmentation\n")
            f.write("   → Verify annotation quality\n")
        
        f.write(f"\n2. Overall model performance:\n")
        if metrics.box.map50 >= 0.5:
            f.write(f"   ✅ Good overall mAP@0.5: {metrics.box.map50:.1%}\n")
        else:
            f.write(f"   ⚠️  Low overall mAP@0.5: {metrics.box.map50:.1%}\n")
            f.write(f"   → Consider training longer (more epochs)\n")
            f.write(f"   → Try larger model (YOLOv8m)\n")
    
    print(f"\n>>>>> Detailed report saved to: {report_path}")
    print(f">>>>> Analysis complete! Check {output_dir} for all outputs")
    
    # Also save to logs directory
    import shutil
    from datetime import datetime
    
    log_cm = log_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    log_report = log_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    shutil.copy(cm_path, log_cm)
    shutil.copy(report_path, log_report)
    
    print(f"\n>>>>> Logs also saved to:")
    print(f"   {log_cm}")
    print(f"   {log_report}")
    
else:
    print("\n----- Confusion matrix not available in metrics")
    print("   This may happen with some YOLO versions or if validation failed")