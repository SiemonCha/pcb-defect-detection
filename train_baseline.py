"""
YOLOv8n Baseline - Fast training to verify everything works
Expected time: 30-60 mins on GPU, 3-4 hours on CPU
"""

from ultralytics import YOLO
import torch

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>>>> Training on: {device.upper()}")

# Load model
model = YOLO('yolov8n.pt')

# Train with minimal settings
results = model.train(
    data='data/printed-circuit-board-2/data.yaml',  # Update path if different
    epochs=50,
    imgsz=640,
    batch=16 if device == 'cuda' else 4,
    device=device,
    project='runs/train',
    name='baseline_yolov8n',
    patience=10,  # Early stopping
    save=True,
    plots=True,
    verbose=True
)

print(f"\n------ Baseline training complete!")
print(f">>>>> Results: runs/train/baseline_yolov8n")
print(f">>>>> Best weights: runs/train/baseline_yolov8n/weights/best.pt")