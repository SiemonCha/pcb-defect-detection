"""
YOLOv8n Baseline - Fast training
"""

from ultralytics import YOLO
import torch

# Check device
if torch.backends.mps.is_available():
    device = 'mps'
    print(f">>>> Training on: Apple Silicon (MPS)")
else:
    device = 'cpu'
    print(f">>>> Training on: CPU")

# Load model
print("==== Loading YOLOv8n...")
model = YOLO('yolov8n.pt')

# Train
print(f"==== Starting training...")
results = model.train(
    data='data/data.yaml',  # ← Fixed path
    epochs=50,
    imgsz=640,
    batch=8,
    device=device,
    project='runs/train',
    name='baseline_yolov8n',
    patience=10,
    save=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=False
)

print(f"\n>>>> Training complete!")
print(f">>>> Results: runs/train/baseline_yolov8n")
print(f">>>> Best: runs/train/baseline_yolov8n/weights/best.pt")