"""
YOLOv8n Baseline - Fast training with multi-platform support
"""

from ultralytics import YOLO
import torch
import platform

def get_device_info():
    """Get device information and capabilities"""
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f">>>> Training on: NVIDIA GPU - {device_name} ({memory:.1f} GB)")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
        device_name = torch.xpu.get_device_name()
        print(f">>>> Training on: AMD GPU - {device_name}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f">>>> Training on: Apple Silicon ({platform.processor()})")
    else:
        device = 'cpu'
        print(f">>>> Training on: CPU ({platform.processor()})")
    return device

# Set device
device = get_device_info()

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