"""
Download PCB Defect Dataset from Roboflow
Run: python scripts/download_dataset.py
"""

from roboflow import Roboflow
import os
from pathlib import Path

# Create data directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(">>>>> Downloading PCB Defect Dataset from Roboflow...")
print("=" * 60)

# Initialize Roboflow (you'll need to create free account)
# Get API key from: https://app.roboflow.com/settings/api
API_KEY = input("Enter your Roboflow API key (or press Enter to get one): ").strip()

if not API_KEY:
    print("\n>>>>> Get your FREE API key:")
    print("1. Go to https://app.roboflow.com/")
    print("2. Sign up (free)")
    print("3. Go to Settings → API")
    print("4. Copy your API key")
    print("\nRun this script again with your API key.")
    exit()

try:
    rf = Roboflow(api_key=API_KEY)
    
    # Access PCB Defects dataset
    # Using Roboflow 100 PCB dataset
    project = rf.workspace("roboflow-100").project("printed-circuit-board")
    
    # Download dataset in YOLOv8 format
    dataset = project.version(2).download("yolov8", location=str(DATA_DIR))
    
    print("\n----- Dataset downloaded successfully!")
    print(f">>>>> Location: {DATA_DIR / 'printed-circuit-board-2'}")
    print("\n>>>>> Dataset structure:")
    print("   ├── train/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   ├── valid/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   ├── test/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── data.yaml")
    
    # Read data.yaml to show class info
    import yaml
    yaml_path = DATA_DIR / "printed-circuit-board-2" / "data.yaml"
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n>>>>> Classes ({data_config['nc']}):")
    for idx, name in enumerate(data_config['names']):
        print(f"   {idx}: {name}")
    
    print(f"\n>>>>> Image counts:")
    for split in ['train', 'valid', 'test']:
        img_dir = DATA_DIR / "printed-circuit-board-2" / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
            print(f"   {split}: {count} images")
    
    # Save dataset path for later use
    with open("data/dataset_path.txt", "w") as f:
        f.write(str(DATA_DIR / "printed-circuit-board-2"))
    
    print("\n✨ Ready for EDA! Run notebooks/01_eda.ipynb")
    
except Exception as e:
    print(f"\nXXXXX Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Ensure you have internet connection")
    print("3. Try accessing https://app.roboflow.com in browser")
    print("\nAlternative: Download manually from:")
    print("https://universe.roboflow.com/roboflow-100/printed-circuit-board")