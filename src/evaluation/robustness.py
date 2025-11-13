"""
Robustness Testing - Test model on corrupted/edge case inputs

Tests model behavior under:
- Noise (Gaussian, salt & pepper)
- Blur (Gaussian, motion)
- Brightness/contrast changes
- Occlusions
- Resolution changes

Usage:
    python -m cli robustness                    # Auto-detect model
    python -m cli robustness --model runs/.../best.pt
    python -m cli robustness --samples 20       # More test samples
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import glob
import argparse
from pathlib import Path
import yaml
from data import resolve_dataset_yaml

def find_best_model():
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    raise FileNotFoundError("No trained model found")

def find_data_yaml() -> Path:
    return resolve_dataset_yaml()

# Corruption functions
def add_gaussian_noise(img, mean=0, std=25):
    """Add Gaussian noise"""
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.01):
    """Add salt and pepper noise"""
    noisy = img.copy()
    h, w = img.shape[:2]
    
    # Salt
    num_salt = int(prob * h * w)
    coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper
    num_pepper = int(prob * h * w)
    coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy

def add_gaussian_blur(img, kernel_size=5):
    """Add Gaussian blur"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def add_motion_blur(img, kernel_size=15):
    """Add motion blur"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def adjust_brightness(img, factor=1.5):
    """Adjust brightness"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def adjust_contrast(img, factor=1.5):
    """Adjust contrast"""
    mean = np.mean(img)
    contrasted = (img - mean) * factor + mean
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def add_occlusion(img, num_blocks=3, block_size=50):
    """Add random black occlusions"""
    occluded = img.copy()
    h, w = img.shape[:2]
    
    for _ in range(num_blocks):
        x = np.random.randint(0, max(1, w - block_size))
        y = np.random.randint(0, max(1, h - block_size))
        occluded[y:y+block_size, x:x+block_size] = 0
    
    return occluded

def resize_low_res(img, scale=0.5):
    """Simulate low resolution"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

# Test corruptions
CORRUPTIONS = {
    'gaussian_noise_light': lambda img: add_gaussian_noise(img, std=15),
    'gaussian_noise_heavy': lambda img: add_gaussian_noise(img, std=40),
    'salt_pepper': lambda img: add_salt_pepper_noise(img, prob=0.02),
    'gaussian_blur': lambda img: add_gaussian_blur(img, kernel_size=7),
    'motion_blur': lambda img: add_motion_blur(img, kernel_size=15),
    'brightness_up': lambda img: adjust_brightness(img, factor=1.5),
    'brightness_down': lambda img: adjust_brightness(img, factor=0.6),
    'contrast_up': lambda img: adjust_contrast(img, factor=1.8),
    'contrast_down': lambda img: adjust_contrast(img, factor=0.5),
    'occlusion': lambda img: add_occlusion(img, num_blocks=3, block_size=50),
    'low_resolution': lambda img: resize_low_res(img, scale=0.5),
}

def test_corruption(model, img_path, corruption_name, corruption_fn):
    """Test model on corrupted image"""
    img_path = Path(img_path)
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get baseline predictions
    results_clean = model.predict(str(img_path), conf=0.25, verbose=False)
    clean_count = len(results_clean[0].boxes) if results_clean[0].boxes is not None else 0
    clean_conf = float(np.mean(results_clean[0].boxes.conf.cpu().numpy())) if clean_count > 0 else 0
    
    # Apply corruption
    img_corrupted = corruption_fn(img)
    
    # Get corrupted predictions
    results_corrupted = model.predict(img_corrupted, conf=0.25, verbose=False)
    corrupted_count = len(results_corrupted[0].boxes) if results_corrupted[0].boxes is not None else 0
    corrupted_conf = float(np.mean(results_corrupted[0].boxes.conf.cpu().numpy())) if corrupted_count > 0 else 0
    
    # Calculate degradation
    count_degradation = (clean_count - corrupted_count) / clean_count if clean_count > 0 else 0
    conf_degradation = (clean_conf - corrupted_conf) / clean_conf if clean_conf > 0 else 0
    
    return {
        'corruption': corruption_name,
        'clean_count': clean_count,
        'corrupted_count': corrupted_count,
        'clean_conf': clean_conf,
        'corrupted_conf': corrupted_conf,
        'count_degradation': count_degradation,
        'conf_degradation': conf_degradation,
        'img_clean': img,
        'img_corrupted': img_corrupted
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--samples', type=int, default=10, help='Number of test images')
    args = parser.parse_args()
    
    # Setup
    model_path = args.model or find_best_model()
    data_yaml = find_data_yaml()
    
    print(f"{'='*60}")
    print("ROBUSTNESS TESTING")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Corruptions: {len(CORRUPTIONS)}")
    print(f"Test samples: {args.samples}")
    
    # Load model
    model = YOLO(model_path)
    
    # Get test images
    with data_yaml.open('r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = data_yaml.parent
    test_img_dir = dataset_root / 'test' / 'images'
    
    test_images = sorted(test_img_dir.glob('*.jpg')) + \
                  sorted(test_img_dir.glob('*.png'))
    
    if not test_images:
        print(f"xxxx No test images found")
        return
    
    # Sample images
    import random
    random.seed(42)
    test_images = random.sample(test_images, min(args.samples, len(test_images)))
    
    print(f">>>> Testing {len(test_images)} images x {len(CORRUPTIONS)} corruptions...")
    
    # Test all corruptions
    results_all = {corruption: [] for corruption in CORRUPTIONS.keys()}
    
    for img_path in test_images:
        for corruption_name, corruption_fn in CORRUPTIONS.items():
            result = test_corruption(model, img_path, corruption_name, corruption_fn)
            if result:
                results_all[corruption_name].append(result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("ROBUSTNESS RESULTS")
    print(f"{'='*60}")
    
    print(f"\n{'Corruption':<25s} {'Detect Loss':>12s} {'Conf Loss':>12s} {'Status':>12s}")
    print("-" * 65)
    
    summary = {}
    for corruption_name in CORRUPTIONS.keys():
        results = results_all[corruption_name]
        
        if results:
            avg_count_deg = np.mean([r['count_degradation'] for r in results])
            avg_conf_deg = np.mean([r['conf_degradation'] for r in results])
            
            # Status
            if avg_count_deg < 0.1 and avg_conf_deg < 0.1:
                status = "Robust"
            elif avg_count_deg < 0.3 and avg_conf_deg < 0.3:
                status = "Moderate"
            else:
                status = "Fragile"
            
            print(f"{corruption_name:<25s} {avg_count_deg:>11.1%} {avg_conf_deg:>11.1%} {status:>12s}")
            
            summary[corruption_name] = {
                'count_degradation': avg_count_deg,
                'conf_degradation': avg_conf_deg,
                'status': status
            }
    
    # Visualize examples
    output_dir = Path('logs/robustness')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>>> Generating visualizations...")
    
    # Visualize worst cases for each corruption
    for corruption_name in list(CORRUPTIONS.keys())[:6]:  # First 6 for space
        results = results_all[corruption_name]
        if not results:
            continue
        
        # Find worst case
        worst = max(results, key=lambda r: r['count_degradation'] + r['conf_degradation'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(worst['img_clean'])
        ax1.set_title(f'Clean ({worst["clean_count"]} detections, conf={worst["clean_conf"]:.2f})', 
                     fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(worst['img_corrupted'])
        ax2.set_title(f'{corruption_name}\n({worst["corrupted_count"]} detections, conf={worst["corrupted_conf"]:.2f})',
                     fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'robustness_{corruption_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    corruptions = list(summary.keys())
    count_degs = [summary[c]['count_degradation'] for c in corruptions]
    conf_degs = [summary[c]['conf_degradation'] for c in corruptions]
    
    # Detection loss
    colors = ['green' if d < 0.1 else 'orange' if d < 0.3 else 'red' for d in count_degs]
    ax1.barh(corruptions, count_degs, color=colors)
    ax1.set_xlabel('Detection Loss', fontsize=12)
    ax1.set_title('Detection Count Degradation', fontsize=13, fontweight='bold')
    ax1.axvline(0.1, color='green', linestyle='--', alpha=0.5, label='Robust (<10%)')
    ax1.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (<30%)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    
    # Confidence loss
    colors = ['green' if d < 0.1 else 'orange' if d < 0.3 else 'red' for d in conf_degs]
    ax2.barh(corruptions, conf_degs, color=colors)
    ax2.set_xlabel('Confidence Loss', fontsize=12)
    ax2.set_title('Confidence Degradation', fontsize=13, fontweight='bold')
    ax2.axvline(0.1, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(0.3, color='orange', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save report
    from datetime import datetime
    report_path = output_dir / f'robustness_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(report_path, 'w') as f:
        f.write("ROBUSTNESS TESTING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Test samples: {len(test_images)}\n")
        f.write(f"Corruptions tested: {len(CORRUPTIONS)}\n\n")
        
        f.write("Results by corruption:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Corruption':<25s} {'Detect Loss':>12s} {'Conf Loss':>12s} {'Status':>12s}\n")
        f.write("-"*60 + "\n")
        
        for corruption_name in CORRUPTIONS.keys():
            if corruption_name in summary:
                s = summary[corruption_name]
                f.write(f"{corruption_name:<25s} {s['count_degradation']:>11.1%} "
                       f"{s['conf_degradation']:>11.1%} {s['status']:>12s}\n")
        
        f.write("\n\nInterpretation:\n")
        f.write("-"*60 + "\n")
        
        robust_count = sum(1 for s in summary.values() if s['status'] == 'Robust')
        moderate_count = sum(1 for s in summary.values() if s['status'] == 'Moderate')
        fragile_count = sum(1 for s in summary.values() if s['status'] == 'Fragile')
        
        f.write(f"Robust corruptions: {robust_count}/{len(summary)}\n")
        f.write(f"Moderate impact: {moderate_count}/{len(summary)}\n")
        f.write(f"Fragile to: {fragile_count}/{len(summary)}\n\n")
        
        if fragile_count > len(summary) * 0.5:
            f.write("Model is not robust to common corruptions\n")
            f.write("   Recommendations:\n")
            f.write("   - Add corruption augmentation during training\n")
            f.write("   - Use stronger data augmentation\n")
            f.write("   - Test on more diverse real-world data\n")
        elif moderate_count > len(summary) * 0.3:
            f.write("Model has moderate robustness\n")
            f.write("   Consider augmentation for fragile cases\n")
        else:
            f.write("Model is robust to tested corruptions\n")
            f.write("   Ready for real-world deployment\n")
    
    print(f"\n>>>> Report: {report_path}")
    print(f">>>> Visualizations: {output_dir}")
    print(f"\n>>>> Summary:")
    print(f"   Robust: {robust_count}/{len(summary)}")
    print(f"   Moderate: {moderate_count}/{len(summary)}")
    print(f"   Fragile: {fragile_count}/{len(summary)}")

if __name__ == '__main__':
    main()
