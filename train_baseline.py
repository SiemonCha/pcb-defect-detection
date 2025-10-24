"""
YOLOv8n Baseline - Fast training with multi-platform support
"""

from ultralytics import YOLO
import torch
import platform
import os
import glob

def get_device_info():
    """Get device information and capabilities"""
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        if torch.version.hip is not None:
            print(f">>>> Training on: AMD GPU (ROCm) - {device_name}")
            return device, True
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f">>>> Training on: NVIDIA GPU - {device_name} ({memory:.1f} GB)")
            return device, False
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f">>>> Training on: Apple Silicon - {platform.processor()}")
        return device, False
    else:
        device = 'cpu'
        print(f">>>> Training on: CPU - {platform.processor()}")
        return device, False

def find_data_yaml():
    """Find data.yaml in data directory"""
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml):
            return data_yaml
    
    patterns = [
        'data/*/data.yaml',
        'data/data.yaml',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    raise FileNotFoundError(
        "data.yaml not found. Run: python data_download.py\n"
        "Expected locations:\n"
        "  - data/Printed-Circuit-Board-2/data.yaml\n"
        "  - data/printed-circuit-board-2/data.yaml"
    )

device, is_rocm = get_device_info()
data_yaml = find_data_yaml()
print(f"==== Using dataset: {data_yaml}")

print("==== Loading YOLOv8n...")
model = YOLO('yolov8n.pt')

print(f"==== Starting training...")
results = model.train(
    data=data_yaml,
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
    cache='ram' if device != 'cpu' else False,
    amp=True if (device == 'cuda' and not is_rocm) else False
)

print(f"\n>>>> Training complete!")
print(f">>>> Results: runs/train/baseline_yolov8n")
print(f">>>> Best: runs/train/baseline_yolov8n/weights/best.pt")
print(f"\n>>>> Next step: python evaluate.py")