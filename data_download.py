"""
PCB Defect Dataset Downloader
Uses Roboflow's official download code
"""
import os
import sys
import yaml
from pathlib import Path

def verify_dataset(dataset_location):
    """Verify downloaded dataset"""
    data_yaml = os.path.join(dataset_location, 'data.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml not found at {data_yaml}")
        return False
    
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"\n{'='*60}")
        print("DATASET VERIFICATION")
        print(f"{'='*60}")
        print(f"Location: {dataset_location}")
        print(f"Classes: {data['nc']}")
        print(f"Names: {data['names']}")
        
        # Check if it's defects (not components)
        defect_keywords = ['missing', 'short', 'spur', 'open', 'hole', 'mouse', 'bite', 'spurious']
        component_keywords = ['resistor', 'capacitor', 'button', 'diode', 'transistor']
        
        names_str = str(data['names']).lower()
        
        is_defect = any(kw in names_str for kw in defect_keywords)
        is_component = any(kw in names_str for kw in component_keywords)
        
        if is_defect and not is_component:
            print(f"\n✅ CORRECT: Defect detection dataset")
        elif is_component:
            print(f"\n❌ WRONG: Component detection dataset")
            print(f"   This won't work for defect detection!")
            return False
        else:
            print(f"\n⚠️  Unknown dataset type - verify manually")
        
        # Count images
        train_imgs = 0
        valid_imgs = 0
        test_imgs = 0
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_location, split, 'images')
            if os.path.exists(img_dir):
                imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if split == 'train':
                    train_imgs = len(imgs)
                elif split == 'valid':
                    valid_imgs = len(imgs)
                else:
                    test_imgs = len(imgs)
        
        print(f"\n{'='*60}")
        print("IMAGE COUNTS")
        print(f"{'='*60}")
        print(f"Train: {train_imgs}")
        print(f"Valid: {valid_imgs}")
        print(f"Test:  {test_imgs}")
        
        if train_imgs == 0:
            print(f"\n❌ No training images found!")
            return False
        
        # Save dataset path
        with open('dataset_path.txt', 'w') as f:
            f.write(dataset_location)
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset path saved to: dataset_path.txt")
        print(f"{'='*60}")
        print(f"\n✅ READY TO TRAIN!")
        print(f"   Next: python train_baseline.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("="*60)
    print("PCB DEFECT DATASET DOWNLOADER")
    print("="*60)
    
    # Check if dataset already exists
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            existing_path = f.read().strip()
        
        if os.path.exists(existing_path):
            print(f"\n✅ Dataset already exists: {existing_path}")
            if verify_dataset(existing_path):
                print("\n💡 Dataset is ready. No need to re-download.")
                choice = input("\nRe-download anyway? (y/N): ").lower().strip()
                if choice != 'y':
                    return
            print("\n==== Re-downloading dataset...")
    
    # Download using Roboflow
    print("\n==== Downloading from Roboflow...")
    
    try:
        # Check if roboflow is installed
        try:
            from roboflow import Roboflow
        except ImportError:
            print("\n❌ Roboflow package not installed")
            print("   Installing roboflow...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
            from roboflow import Roboflow
            print("   ✅ Roboflow installed")
        
        # Roboflow download code (from "Show download code")
        print("   Connecting to Roboflow...")
        rf = Roboflow(api_key="NdpJQ5HepZGa7Xvw9wQD")
        project = rf.workspace("uni-4sdfm").project("pcb-defects")
        version = project.version(2)
        
        print("   Downloading dataset...")
        dataset = version.download("yolov8")
        
        print(f"\n✅ Download complete!")
        print(f"   Location: {dataset.location}")
        
        # Verify
        if verify_dataset(dataset.location):
            print("\n" + "="*60)
            print("✅✅✅ SUCCESS - Dataset ready for training ✅✅✅")
            print("="*60)
        else:
            print("\n⚠️  Dataset downloaded but verification failed")
            print("   Check the dataset manually")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Run: pip install roboflow")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("\n1. Check internet connection")
        print("2. Verify API key is correct")
        print("3. Try manual download:")
        print("   - Visit: https://universe.roboflow.com/uni-4sdfm/pcb-defects")
        print("   - Click 'Download' → Select 'YOLOv8'")
        print("   - Extract to: pcb-defect-detection/data/")
        print("   - Run: python data_download.py")
        print("\n" + "="*60)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)