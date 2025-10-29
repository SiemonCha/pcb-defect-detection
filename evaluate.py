"""
Evaluate trained model on test set with automatic logging

Usage:
    python evaluate.py                           # Auto-detect best model
    python evaluate.py runs/.../weights/best.pt  # Specify model
"""

from ultralytics import YOLO
import os
import sys
import glob
import json
from datetime import datetime
from pathlib import Path

# CLI argument or auto-detect
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
else:
    # Auto-detect: prefer production > baseline
    model_patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(pattern)
        if matches:
            model_path = max(matches, key=os.path.getmtime)
            break
    
    if not model_path:
        raise FileNotFoundError(
            "No trained model found. Train first:\n"
            "  python train_baseline.py\n"
            "  python train_production.py\n"
            f"Checked: {model_patterns}"
        )

# Get dataset path
if os.path.exists('dataset_path.txt'):
    with open('dataset_path.txt', 'r') as f:
        dataset_location = f.read().strip()
    data_yaml = os.path.join(dataset_location, 'data.yaml')
else:
    yaml_matches = glob.glob('data/*/data.yaml') + glob.glob('data/data.yaml')
    if yaml_matches:
        data_yaml = yaml_matches[0]
    else:
        data_yaml = 'data/data.yaml'

if not os.path.exists(data_yaml):
    raise FileNotFoundError(
        f"Dataset not found: {data_yaml}\n"
        "Run: python data_download.py"
    )

# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Load model
model = YOLO(model_path)
print(f">>>>> Evaluating: {model_path}")
print(f">>>>> Dataset: {data_yaml}")

# Run validation on test set
metrics = model.val(
    data=data_yaml,
    split='test',
    plots=True,
    save_json=True
)

# Prepare results
results = {
    'timestamp': datetime.now().isoformat(),
    'model_path': model_path,
    'model_name': os.path.basename(model_path),
    'dataset': data_yaml,
    'metrics': {
        'mAP@0.5': float(metrics.box.map50),
        'mAP@0.5:0.95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
    },
    'per_class': {},
    'speed': {}
}

# Per-class metrics
if hasattr(metrics, 'names') and hasattr(metrics.box, 'ap50'):
    for name, ap in zip(metrics.names.values(), metrics.box.ap50):
        results['per_class'][name] = {
            'AP@0.5': float(ap)
        }

# Speed metrics
if hasattr(metrics, 'speed') and metrics.speed:
    for key, val in metrics.speed.items():
        if isinstance(val, (int, float)):
            results['speed'][key] = float(val)

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")

# Per-class metrics
if results['per_class']:
    print("\n>>>>> Per-Class Performance:")
    print("-" * 60)
    for name, perf in results['per_class'].items():
        print(f"{name:20s} | AP@0.5: {perf['AP@0.5']:.4f}")

# Speed
if results['speed']:
    print(f"\n⚡ Inference Speed:")
    total_time = 0
    for key, val in results['speed'].items():
        print(f"{key.capitalize():12s}: {val:.1f}ms")
        total_time += val
    print(f"{'Total':12s}: {total_time:.1f}ms")
    results['speed']['total'] = total_time
else:
    total_time = 0

# Save to JSON log
json_log = log_dir / 'evaluation_log.json'
if json_log.exists():
    with open(json_log, 'r') as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(results)

with open(json_log, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n>>>>> Results logged to: {json_log}")

# Save summary report
report_file = log_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, 'w') as f:
    f.write("EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {results['timestamp']}\n")
    f.write(f"Model: {results['model_name']}\n")
    f.write(f"Dataset: {data_yaml}\n\n")
    
    f.write("Overall Metrics:\n")
    f.write("-" * 60 + "\n")
    f.write(f"mAP@0.5:      {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
    f.write(f"Precision:    {metrics.box.mp:.4f}\n")
    f.write(f"Recall:       {metrics.box.mr:.4f}\n\n")
    
    if results['per_class']:
        f.write("Per-Class Performance:\n")
        f.write("-" * 60 + "\n")
        for name, perf in results['per_class'].items():
            f.write(f"{name:20s} | AP@0.5: {perf['AP@0.5']:.4f}\n")
        f.write("\n")
    
    if results['speed']:
        f.write("Inference Speed:\n")
        f.write("-" * 60 + "\n")
        for key, val in results['speed'].items():
            f.write(f"{key.capitalize():12s}: {val:.1f}ms\n")
        f.write("\n")
    
    f.write("Performance Assessment:\n")
    f.write("-" * 60 + "\n")
    
    if metrics.box.map50 > 0.85:
        f.write("✅ PASS: Model meets 85% mAP@0.5 target\n")
    else:
        f.write(f"⚠️  WARNING: mAP@0.5 is {metrics.box.map50:.1%}, target is 85%\n")
        f.write("   Recommendation: Train YOLOv8m or increase epochs\n")
    
    if total_time > 0 and total_time < 100:
        f.write("✅ PASS: Inference < 100ms target\n")
    elif total_time > 0:
        f.write(f"⚠️  WARNING: Inference is {total_time:.0f}ms\n")
        f.write("   Recommendation: Use YOLOv8n or optimize to ONNX\n")

print(f">>>>> Report saved to: {report_file}")

# Check targets
print("\n" + "="*60)
if metrics.box.map50 > 0.85:
    print(">>>>> PASS: Model meets 85% mAP@0.5 target")
else:
    print(f">>>>> WARNING: mAP@0.5 is {metrics.box.map50:.1%}, target is 85%")
    print("   → Consider training YOLOv8m or increasing epochs")

if total_time > 0 and total_time < 100:
    print(">>>>> PASS: Inference < 100ms target")
elif total_time > 0:
    print(f">>>>> WARNING: Inference is {total_time:.0f}ms")
    print("   → Consider using YOLOv8n or optimizing to ONNX")

print("\n>>>>> All results automatically logged to logs/ directory")
print(f">>>>> View logs at: {log_dir.absolute()}")