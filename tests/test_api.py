"""
Test client for PCB Defect Detection API

Usage:
    python test_api.py                           # Test with sample image
    python test_api.py --image path/to/image.jpg # Test with custom image
    python test_api.py --url http://localhost:8000 # Custom API URL
"""

import requests
import argparse
import os
import json
from pathlib import Path

def test_health(api_url):
    """Test health endpoint"""
    print(f"\n{'='*60}")
    print("Testing /health endpoint...")
    print(f"{'='*60}")
    
    response = requests.get(f"{api_url}/health")
    
    if response.status_code == 200:
        print("✅ Health check passed")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(response.text)
    
    return response.status_code == 200

def test_model_info(api_url):
    """Test model-info endpoint"""
    print(f"\n{'='*60}")
    print("Testing /model-info endpoint...")
    print(f"{'='*60}")
    
    response = requests.get(f"{api_url}/model-info")
    
    if response.status_code == 200:
        print("✅ Model info retrieved")
        data = response.json()
        print(f"\nModel: {data['model_type']}")
        print(f"Classes ({data['num_classes']}): {', '.join(data['class_names'])}")
    else:
        print(f"❌ Model info failed: {response.status_code}")
        print(response.text)
    
    return response.status_code == 200

def test_detection(api_url, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Test detection endpoint"""
    print(f"\n{'='*60}")
    print("Testing /detect endpoint...")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Prepare request
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        params = {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        response = requests.post(
            f"{api_url}/detect",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Detection successful")
        print(f"   Inference time: {data['inference_time_ms']:.2f}ms")
        print(f"   Image size: {data['image_size']}")
        print(f"   Defects found: {data['count']}")
        
        if data['count'] > 0:
            print(f"\n   Detected defects:")
            for i, det in enumerate(data['detections'], 1):
                print(f"   {i}. {det['class_name']}: {det['confidence']:.2%} @ {det['bbox']}")
        else:
            print(f"   No defects detected (threshold: {conf_threshold})")
        
        return True
    else:
        print(f"❌ Detection failed: {response.status_code}")
        print(response.text)
        return False

def find_sample_image():
    """Find a sample image from dataset"""
    patterns = [
        'data/*/test/images/*.jpg',
        'data/*/valid/images/*.jpg',
        'data/*/train/images/*.jpg',
    ]
    
    import glob
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Test PCB Defect Detection API")
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API URL')
    parser.add_argument('--image', type=str, default=None, help='Path to test image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PCB DEFECT DETECTION API - TEST CLIENT")
    print(f"{'='*60}")
    print(f"API URL: {args.url}")
    
    # Test health
    if not test_health(args.url):
        print("\n❌ API is not healthy. Make sure the server is running:")
        print("   python api.py")
        return
    
    # Test model info
    test_model_info(args.url)
    
    # Test detection
    if args.image:
        image_path = args.image
    else:
        print("\nNo image specified, searching for sample image...")
        image_path = find_sample_image()
        
        if not image_path:
            print("❌ No sample images found. Please specify --image path/to/image.jpg")
            return
        
        print(f"Using sample image: {image_path}")
    
    test_detection(args.url, image_path, args.conf, args.iou)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()