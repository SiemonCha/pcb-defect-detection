"""
Production Monitoring - Track deployed model performance

Monitors:
- Inference latency
- Confidence distribution
- Detection rate changes
- Error rates
- Resource usage

Usage:
    # Start monitoring server
    python -m cli monitor --api-url http://localhost:8000

    # Generate report from logs
    python -m cli monitor --analyze logs/inference_log.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

class InferenceMonitor:
    def __init__(self, log_file='logs/inference_log.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'timestamps': [],
            'latencies': [],
            'confidences': [],
            'detection_counts': [],
            'errors': [],
            'image_sizes': []
        }
    
    def log_inference(self, latency, detections, image_size, error=None):
        """Log a single inference"""
        timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'latency_ms': latency,
            'num_detections': len(detections) if detections else 0,
            'image_size': image_size,
            'error': error
        }
        
        if detections:
            confs = [d['confidence'] for d in detections]
            entry['avg_confidence'] = float(np.mean(confs))
            entry['min_confidence'] = float(np.min(confs))
            entry['max_confidence'] = float(np.max(confs))
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        return entry
    
    def load_logs(self):
        """Load logs from file"""
        if not self.log_file.exists():
            return []
        
        logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
        
        return logs
    
    def analyze_logs(self, logs):
        """Analyze collected logs"""
        if not logs:
            return None
        
        latencies = [l['latency_ms'] for l in logs if 'latency_ms' in l]
        detection_counts = [l['num_detections'] for l in logs if 'num_detections' in l]
        confidences = [l.get('avg_confidence', 0) for l in logs if 'avg_confidence' in l]
        errors = [l for l in logs if l.get('error')]
        
        analysis = {
            'total_inferences': len(logs),
            'time_period': {
                'start': logs[0]['timestamp'],
                'end': logs[-1]['timestamp']
            },
            'latency': {
                'mean': np.mean(latencies) if latencies else 0,
                'std': np.std(latencies) if latencies else 0,
                'p50': np.percentile(latencies, 50) if latencies else 0,
                'p95': np.percentile(latencies, 95) if latencies else 0,
                'p99': np.percentile(latencies, 99) if latencies else 0,
                'min': np.min(latencies) if latencies else 0,
                'max': np.max(latencies) if latencies else 0
            },
            'detection_rate': {
                'mean': np.mean(detection_counts) if detection_counts else 0,
                'std': np.std(detection_counts) if detection_counts else 0,
                'zero_detection_rate': sum(1 for d in detection_counts if d == 0) / len(detection_counts) if detection_counts else 0
            },
            'confidence': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': np.min(confidences) if confidences else 0,
                'max': np.max(confidences) if confidences else 0
            },
            'errors': {
                'count': len(errors),
                'rate': len(errors) / len(logs) if logs else 0
            }
        }
        
        return analysis

def test_api_endpoint(api_url, test_image):
    """Test API with a sample image"""
    try:
        start = time.perf_counter()
        
        with open(test_image, 'rb') as f:
            response = requests.post(
                f"{api_url}/detect",
                files={'file': f},
                timeout=30
            )
        
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return {
                'latency': latency,
                'detections': data.get('detections', []),
                'image_size': data.get('image_size'),
                'error': None
            }
        else:
            return {
                'latency': latency,
                'detections': None,
                'image_size': None,
                'error': f"HTTP {response.status_code}"
            }
    
    except Exception as e:
        return {
            'latency': None,
            'detections': None,
            'image_size': None,
            'error': str(e)
        }

def monitor_live(api_url, test_images, duration_sec=300, interval_sec=10):
    """Monitor API for specified duration"""
    print(f"{'='*60}")
    print("LIVE MONITORING")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"Duration: {duration_sec}s")
    print(f"Interval: {interval_sec}s")
    print(f"Test images: {len(test_images)}")
    
    monitor = InferenceMonitor()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while (time.time() - start_time) < duration_sec:
            iteration += 1
            
            # Select random test image
            import random
            test_image = random.choice(test_images)
            
            # Test endpoint
            result = test_api_endpoint(api_url, test_image)
            
            # Log
            monitor.log_inference(
                latency=result['latency'] or 0,
                detections=result['detections'],
                image_size=result['image_size'],
                error=result['error']
            )
            
            # Print status
            if result['error']:
                print(f"[{iteration}] ERROR [{iteration}]: {result['error']}")
            else:
                print(f"[{iteration}] OK    latency {result['latency']:.1f}ms, "
                     f"{len(result['detections'])} detections")
            
            # Wait
            time.sleep(interval_sec)
    
    except KeyboardInterrupt:
        print(f"\n>>>> Monitoring stopped by user")
    
    print(f"\n>>>> Collected {iteration} samples")
    print(f">>>> Log file: {monitor.log_file}")
    
    return monitor.log_file

def generate_report(log_file):
    """Generate monitoring report"""
    monitor = InferenceMonitor(log_file)
    logs = monitor.load_logs()
    
    if not logs:
        print(f"xxxx No logs found in {log_file}")
        return
    
    print(f"{'='*60}")
    print("MONITORING REPORT")
    print(f"{'='*60}")
    print(f"Log file: {log_file}")
    print(f"Total inferences: {len(logs)}")
    
    # Analyze
    analysis = monitor.analyze_logs(logs)
    
    print(f"\nTime period:")
    print(f"  Start: {analysis['time_period']['start']}")
    print(f"  End: {analysis['time_period']['end']}")
    
    print(f"\nLatency (ms):")
    print(f"  Mean: {analysis['latency']['mean']:.2f} +/- {analysis['latency']['std']:.2f}")
    print(f"  P50: {analysis['latency']['p50']:.2f}")
    print(f"  P95: {analysis['latency']['p95']:.2f}")
    print(f"  P99: {analysis['latency']['p99']:.2f}")
    print(f"  Range: [{analysis['latency']['min']:.2f}, {analysis['latency']['max']:.2f}]")
    
    print(f"\nDetection rate:")
    print(f"  Mean: {analysis['detection_rate']['mean']:.2f} +/- {analysis['detection_rate']['std']:.2f}")
    print(f"  Zero detections: {analysis['detection_rate']['zero_detection_rate']:.1%}")
    
    print(f"\nConfidence:")
    print(f"  Mean: {analysis['confidence']['mean']:.3f} +/- {analysis['confidence']['std']:.3f}")
    print(f"  Range: [{analysis['confidence']['min']:.3f}, {analysis['confidence']['max']:.3f}]")
    
    print(f"\nErrors:")
    print(f"  Count: {analysis['errors']['count']}")
    print(f"  Rate: {analysis['errors']['rate']:.2%}")
    
    # Visualizations
    output_dir = Path('logs/monitoring')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Latency over time
    ax = axes[0, 0]
    latencies = [l['latency_ms'] for l in logs if 'latency_ms' in l]
    ax.plot(range(len(latencies)), latencies, alpha=0.7)
    ax.axhline(analysis['latency']['mean'], color='r', linestyle='--', 
              label=f"Mean: {analysis['latency']['mean']:.1f}ms")
    ax.axhline(analysis['latency']['p95'], color='orange', linestyle='--',
              label=f"P95: {analysis['latency']['p95']:.1f}ms")
    ax.set_xlabel('Inference #', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Latency Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Latency distribution
    ax = axes[0, 1]
    ax.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(analysis['latency']['p95'], color='r', linestyle='--',
              label=f"P95: {analysis['latency']['p95']:.1f}ms")
    ax.set_xlabel('Latency (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Latency Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Detection counts
    ax = axes[1, 0]
    detection_counts = [l['num_detections'] for l in logs if 'num_detections' in l]
    ax.plot(range(len(detection_counts)), detection_counts, alpha=0.7)
    ax.axhline(analysis['detection_rate']['mean'], color='r', linestyle='--',
              label=f"Mean: {analysis['detection_rate']['mean']:.1f}")
    ax.set_xlabel('Inference #', fontsize=11)
    ax.set_ylabel('Number of Detections', fontsize=11)
    ax.set_title('Detections Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confidence distribution
    ax = axes[1, 1]
    confidences = [l.get('avg_confidence', 0) for l in logs if 'avg_confidence' in l]
    if confidences:
        ax.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0.25, color='r', linestyle='--', label='Threshold (0.25)')
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_path = output_dir / f'monitoring_report_{timestamp}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save text report
    report_path = output_dir / f'monitoring_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("PRODUCTION MONITORING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total inferences: {analysis['total_inferences']}\n")
        f.write(f"Time period: {analysis['time_period']['start']} to {analysis['time_period']['end']}\n\n")
        
        f.write("Latency Statistics (ms):\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean: {analysis['latency']['mean']:.2f} +/- {analysis['latency']['std']:.2f}\n")
        f.write(f"P50: {analysis['latency']['p50']:.2f}\n")
        f.write(f"P95: {analysis['latency']['p95']:.2f}\n")
        f.write(f"P99: {analysis['latency']['p99']:.2f}\n\n")
        
        f.write("Detection Rate:\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean: {analysis['detection_rate']['mean']:.2f} +/- {analysis['detection_rate']['std']:.2f}\n")
        f.write(f"Zero detection rate: {analysis['detection_rate']['zero_detection_rate']:.1%}\n\n")
        
        f.write("Error Rate:\n")
        f.write("-"*60 + "\n")
        f.write(f"Errors: {analysis['errors']['count']}/{analysis['total_inferences']}\n")
        f.write(f"Rate: {analysis['errors']['rate']:.2%}\n\n")
        
        f.write("Performance Assessment:\n")
        f.write("-"*60 + "\n")
        
        # SLA checks
        if analysis['latency']['p95'] < 100:
            f.write("Latency P95 < 100ms (excellent)\n")
        elif analysis['latency']['p95'] < 200:
            f.write("Latency P95 < 200ms (good)\n")
        else:
            f.write("Latency P95 > 200ms (consider optimisation)\n")
        
        if analysis['errors']['rate'] < 0.01:
            f.write("Error rate < 1% (excellent)\n")
        elif analysis['errors']['rate'] < 0.05:
            f.write("Error rate < 5% (acceptable)\n")
        else:
            f.write("Error rate > 5% (needs investigation)\n")
        
        if analysis['detection_rate']['zero_detection_rate'] < 0.1:
            f.write("Low zero-detection rate (< 10%)\n")
        else:
            f.write("High zero-detection rate (> 10%)\n")
            f.write("   Check if model confidence threshold is too high\n")
    
    print(f"\n>>>> Report saved: {report_path}")
    print(f">>>> Visualization: {viz_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                       help='API URL to monitor')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze existing log file')
    parser.add_argument('--duration', type=int, default=300,
                       help='Monitoring duration (seconds)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Test interval (seconds)')
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing logs
        generate_report(Path(args.analyze))
    else:
        # Live monitoring
        # Find test images
        try:
            from pathlib import Path
            import glob
            
            if os.path.exists('dataset_path.txt'):
                with open('dataset_path.txt', 'r') as f:
                    dataset_path = f.read().strip()
                test_img_dir = os.path.join(dataset_path, 'test', 'images')
            else:
                test_img_dir = 'data/*/test/images'
            
            test_images = glob.glob(os.path.join(test_img_dir, '*.jpg')) + \
                         glob.glob(os.path.join(test_img_dir, '*.png'))
            
            if not test_images:
                print(f"xxxx No test images found")
                print(f"   Provide test images or use --analyze to analyze existing logs")
                return
            
            # Start monitoring
            log_file = monitor_live(
                args.api_url,
                test_images,
                args.duration,
                args.interval
            )
            
            # Generate report
            print(f"\n>>>> Generating report...")
            generate_report(log_file)
            
        except Exception as e:
            print(f"xxxx Monitoring failed: {e}")
            print(f"\nUsage:")
            print(f"  1. Start API: python api.py")
            print(f"  2. Monitor: python -m cli monitor")
            print(f"  3. Or analyze logs: python -m cli monitor --analyze logs/inference_log.json")

if __name__ == '__main__':
    main()
