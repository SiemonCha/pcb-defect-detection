"""
Transfer Learning - Fine-tune model on new PCB types

Usage:
    python transfer_learning.py --data path/to/new_data.yaml
    python transfer_learning.py --data path/to/new_data.yaml --base-model runs/.../best.pt
    python transfer_learning.py --data path/to/new_data.yaml --epochs 30

This script allows you to:
1. Fine-tune existing model on new PCB board types
2. Adapt to different defect types or lighting conditions
3. Quick training with small datasets (<100 images)
"""

from ultralytics import YOLO
import torch
import platform
import argparse
import os
import glob

def get_device_info():
    """Get device information"""
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        if torch.version.hip is not None:
            print(f">>>> Transfer learning on: AMD GPU (ROCm) - {device_name}")
            return device, True
        else:
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f">>>> Transfer learning on: NVIDIA GPU - {device_name} ({memory:.1f} GB)")
            return device, False
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f">>>> Transfer learning on: Apple Silicon - {platform.processor()}")
        return device, False
    else:
        device = 'cpu'
        print(f">>>> Transfer learning on: CPU - {platform.processor()}")
        return device, False

def find_base_model():
    """Find the best trained model to use as base"""
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    
    return None

def validate_dataset(data_yaml):
    """Basic validation of dataset"""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset not found: {data_yaml}")
    
    print(f"==== Dataset: {data_yaml}")
    
    # Check if it's a valid YAML file
    import yaml
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in data.yaml: {key}")
        
        print(f">>>>> Classes: {data['nc']}")
        print(f">>>>> Names: {data['names']}")
        
        return True
    except Exception as e:
        raise ValueError(f"Invalid data.yaml: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for PCB Defect Detection")
    parser.add_argument('--data', type=str, required=True, help='Path to new dataset data.yaml')
    parser.add_argument('--base-model', type=str, default=None, help='Base model to fine-tune (default: auto-detect)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--freeze', type=int, default=10, help='Freeze backbone for N epochs (default: 10)')
    parser.add_argument('--name', type=str, default='transfer_learning', help='Project name')
    
    args = parser.parse_args()
    
    # Validate dataset
    validate_dataset(args.data)
    
    # Get or find base model
    if args.base_model:
        base_model_path = args.base_model
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
    else:
        base_model_path = find_base_model()
        if not base_model_path:
            print("\n----- No trained model found. Using pretrained YOLOv8n")
            base_model_path = 'yolov8n.pt'
    
    print(f"==== Base model: {base_model_path}")
    
    # Get device
    device, is_rocm = get_device_info()
    
    # Load model
    print(f"\n==== Loading base model...")
    model = YOLO(base_model_path)
    
    # Start transfer learning
    print(f"\n==== Starting transfer learning...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Freeze backbone: {args.freeze} epochs")
    print()
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        project='runs/transfer',
        name=args.name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        # Transfer learning specific settings
        lr0=args.lr,
        lrf=0.01,  # Final LR = lr0 * lrf
        warmup_epochs=3,
        freeze=args.freeze,  # Freeze backbone layers
        # Light augmentation (small datasets)
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,  # Reduced mosaic
        mixup=0.0,
        # Other settings
        cache='ram' if device != 'cpu' else False,
        amp=True if (device == 'cuda' and not is_rocm) else False,
        optimizer='AdamW',
        cos_lr=True,
    )
    
    print(f"\n>>>> Transfer learning complete!")
    print(f">>>> Results: runs/transfer/{args.name}")
    print(f">>>> Best model: runs/transfer/{args.name}/weights/best.pt")
    print(f"\n>>>> Evaluate your fine-tuned model:")
    print(f"   python evaluate.py runs/transfer/{args.name}/weights/best.pt")
    print(f"\n>>>> Deploy via API:")
    print(f"   python api.py --model runs/transfer/{args.name}/weights/best.pt")

if __name__ == '__main__':
    main()