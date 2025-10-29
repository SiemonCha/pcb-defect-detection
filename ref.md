# Quick Reference Guide

## New Features Added

### 1. ONNX Export (`export_onnx.py`)

**What it does**: Converts your trained PyTorch model to ONNX format, making it 3-5x faster

**Why it matters**: Industrial systems need fast inference. ONNX optimizes the model for production deployment.

**Usage**:

```bash
python export_onnx.py
# Auto-detects best model and benchmarks speed
```

**Output**:

- `best.onnx` - Optimized model
- `benchmark_results.txt` - Speed comparison report
- Console shows exact speedup (e.g., "4.2x faster")

**When to use**: After training, before deployment

---

### 2. Confusion Matrix Analysis (`confusion_matrix.py`)

**What it does**: Shows you exactly which defect types your model struggles with

**Why it matters**: Knowing your model's weaknesses lets you fix them (more data, better augmentation, etc.)

**Usage**:

```bash
python confusion_matrix.py
```

**Output**:

- `confusion_matrix.png` - Visual heatmap
- `performance_report.txt` - Detailed per-class metrics
- Identifies: Low recall (missed defects), Low precision (false alarms), Class imbalance

**Key metrics**:

- **Precision**: How many detections are correct (low = too many false alarms)
- **Recall**: How many actual defects found (low = missing defects)
- **F1**: Balance of precision and recall

**When to use**: After evaluation, to understand model performance

---

### 3. REST API (`api.py`)

**What it does**: Turns your model into a web service that other apps can use

**Why it matters**: Production systems need APIs. This is how you'd integrate into manufacturing lines.

**Usage**:

```bash
# Start server
python api.py

# Test it
python test_api.py
```

**Endpoints**:

- `POST /detect` - Upload image, get defects
- `GET /health` - Check if API is running
- `GET /model-info` - Get model details

**API docs**: Open http://localhost:8000/docs in browser for interactive testing

**When to use**: For demo or actual deployment

---

### 4. Transfer Learning (`transfer_learning.py`)

**What it does**: Fine-tunes your model on new PCB types with minimal data

**Why it matters**: Every factory has different PCBs. Transfer learning lets you adapt quickly with <100 images.

**Usage**:

```bash
python transfer_learning.py --data new_pcb/data.yaml --epochs 30
```

**Key parameters**:

- `--freeze 10` - Freezes backbone for first 10 epochs (prevents overfitting)
- `--lr 0.001` - Lower learning rate than training from scratch
- `--epochs 30-50` - Usually needs less epochs than full training

**When to use**: When you have a new PCB type or different lighting conditions

---

## Typical Workflow

### For Your Project Demo:

1. **Train**: `python train_baseline.py`
2. **Analyze**: `python confusion_matrix.py` (shows critical thinking)
3. **Optimize**: `python export_onnx.py` (shows deployment knowledge)
4. **Deploy**: `python api.py` (shows full-stack capability)

### For CV/Portfolio:

1. Show confusion matrix → "I identified the model struggles with X defect"
2. Show ONNX speedup → "Optimized inference by 4x for production"
3. Demo API → "Deployed as REST API for integration"
4. Show transfer learning → "Can adapt to new PCB types with minimal data"

---

## Key Points for Project Report

**Problem Solving**:

- Used confusion matrix to identify weak classes
- Applied transfer learning for domain adaptation
- Optimized with ONNX for real-time requirements

**Engineering Skills**:

- REST API for production deployment
- Proper error handling and validation
- Multi-platform GPU support

**Critical Thinking**:

- Analyzed per-class performance, not just overall accuracy
- Identified trade-offs (speed vs accuracy)
- Provided actionable recommendations

---

## Common Questions

**Q: Which model should I use?**
A: YOLOv8n for demos (fast), YOLOv8s for accuracy claims

**Q: What's a good mAP@0.5 score?**
A: >80% is acceptable, >85% is good, >90% is excellent

**Q: Why use ONNX?**
A: Industrial systems need <100ms inference. ONNX gets you there.

**Q: What if confusion matrix shows problems?**
A: That's good! It shows you understand the model's limitations. Discuss in report.

**Q: Do I need all 4 features?**
A: No, but:

- **Minimum**: confusion_matrix + export_onnx (shows analysis + optimization)
- **Better**: Add api (shows deployment skills)
- **Best**: All 4 (shows complete ML engineering workflow)

---

## File Outputs

After running everything, you'll have:

```
runs/
├── train/
│   ├── baseline_yolov8n/
│   │   ├── weights/best.pt
│   │   └── weights/best.onnx         # From export_onnx.py
│   └── analysis/
│       ├── confusion_matrix.png      # From confusion_matrix.py
│       └── performance_report.txt    # From confusion_matrix.py
└── transfer/
    └── transfer_learning/            # From transfer_learning.py
        └── weights/best.pt
```

**For report/presentation**:

- Confusion matrix image (shows analysis)
- Benchmark results (shows optimization)
- API screenshot (shows deployment)

---

## Tips

1. **Run confusion matrix first** - It tells you what to fix
2. **Export ONNX before API** - API can use either, but ONNX is faster
3. **Test API locally** - Use test_api.py before showing anyone
4. **Transfer learning last** - Only if you have time or different dataset

Good luck with your year 3 project! 🚀
