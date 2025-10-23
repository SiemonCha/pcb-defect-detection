"""
Evaluate trained model on test set
"""

from ultralytics import YOLO
import pandas as pd

# Load best model
model_path = 'runs/train/production_yolov8s/weights/best.pt'
model = YOLO(model_path)

print(f">>>>> Evaluating: {model_path}")

# Run validation on test set
metrics = model.val(
    data='data/printed-circuit-board-2/data.yaml',
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
print("\n>>>>> Per-Class Performance:")
print("-" * 60)
for i, (name, ap) in enumerate(zip(metrics.names.values(), metrics.box.ap50)):
    print(f"{name:20s} | AP@0.5: {ap:.4f}")

# Speed
print(f"\n⚡ Inference Speed:")
print(f"Preprocess:  {metrics.speed['preprocess']:.1f}ms")
print(f"Inference:   {metrics.speed['inference']:.1f}ms")
print(f"Postprocess: {metrics.speed['postprocess']:.1f}ms")
print(f"Total:       {sum(metrics.speed.values()):.1f}ms")

# Check if meets requirements
print("\n" + "="*60)
if metrics.box.map50 > 0.85:
    print(">>>>> PASS: Model meets 85% mAP@0.5 target")
else:
    print(f">>>>> WARNING: mAP@0.5 is {metrics.box.map50:.1%}, target is 85%")
    print("   → Consider training YOLOv8m or increasing epochs")

if sum(metrics.speed.values()) < 100:
    print(">>>>> PASS: Inference < 100ms target")
else:
    print(f">>>>> WARNING: Inference is {sum(metrics.speed.values()):.0f}ms")
    print("   → Consider using YOLOv8n or optimizing to ONNX")