"""
Compare all trained models side-by-side
"""

from ultralytics import YOLO
import pandas as pd
from pathlib import Path

models = {
    'YOLOv8n': 'runs/train/baseline_yolov8n/weights/best.pt',
    'YOLOv8s': 'runs/train/production_yolov8s/weights/best.pt',
}

results = []

for name, path in models.items():
    if not Path(path).exists():
        print(f">>>>>  {name} not found, skipping...")
        continue
    
    print(f">>>>> Evaluating {name}...")
    model = YOLO(path)
    metrics = model.val(data='data/printed-circuit-board-2/data.yaml', verbose=False)
    
    results.append({
        'Model': name,
        'mAP@0.5': f"{metrics.box.map50:.4f}",
        'mAP@0.5:0.95': f"{metrics.box.map:.4f}",
        'Precision': f"{metrics.box.mp:.4f}",
        'Recall': f"{metrics.box.mr:.4f}",
        'Inference (ms)': f"{metrics.speed['inference']:.1f}",
        'Size (MB)': f"{Path(path).stat().st_size / 1024 / 1024:.1f}"
    })

# Display comparison
df = pd.DataFrame(results)
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Recommendation
best_map = df.loc[df['mAP@0.5'].astype(float).idxmax()]
fastest = df.loc[df['Inference (ms)'].astype(float).idxmin()]

print(f"\n>>>>> Best Accuracy: {best_map['Model']} (mAP@0.5: {best_map['mAP@0.5']})")
print(f"⚡ Fastest: {fastest['Model']} ({fastest['Inference (ms)']}ms)")