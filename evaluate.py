"""
Evaluate trained model on test set

Usage:
    python evaluate.py                           # Auto-detect best model
    python evaluate.py runs/.../weights/best.pt  # Specify model
"""

from ultralytics import YOLO
import os
import sys
import glob

# CLI argument or auto-detect
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
else:
    # Auto-detect: prefer production > baseline
    # Use glob to handle auto-incremented names (baseline_yolov8n, baseline_yolov8n2, etc.)
    model_patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    model_path = None
    for pattern in model_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Get most recent if multiple matches
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
    # Search for data.yaml
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

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")

# Per-class metrics
if hasattr(metrics, 'names') and hasattr(metrics.box, 'ap50'):
    print("\n>>>>> Per-Class Performance:")
    print("-" * 60)
    for name, ap in zip(metrics.names.values(), metrics.box.ap50):
        print(f"{name:20s} | AP@0.5: {ap:.4f}")

# Speed
if hasattr(metrics, 'speed') and metrics.speed:
    print(f"\n⚡ Inference Speed:")
    for key, val in metrics.speed.items():
        if isinstance(val, (int, float)):
            print(f"{key.capitalize():12s}: {val:.1f}ms")
    
    speed_values = [v for v in metrics.speed.values() if isinstance(v, (int, float))]
    if speed_values:
        print(f"{'Total':12s}: {sum(speed_values):.1f}ms")
        total_time = sum(speed_values)
    else:
        total_time = 0
else:
    total_time = 0

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