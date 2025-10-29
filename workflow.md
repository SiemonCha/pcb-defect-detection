# Workflow Guide

## Complete Workflow (First Time)

```bash
# 1. Install dependencies
python install.py

# 2. Download dataset
python data_download.py

# 3. Train model (15-30 min on GPU)
python train_baseline.py

# 4. Analyze trained model (generates all reports)
python auto_analyze.py

# 5. Check results
ls logs/
```

## What Each Script Does

### install.py

- Installs all dependencies
- Detects GPU type
- Installs correct PyTorch version
- **Run once** at setup

### data_download.py

- Downloads PCB dataset from Roboflow
- Saves to `data/` folder
- Creates `dataset_path.txt`
- **Run once** per project

### train_baseline.py

- Trains YOLOv8n model (fast, 50 epochs)
- Saves to `runs/train/baseline_yolov8n/`
- Takes 15-30 min on GPU, longer on CPU
- **Run once** to get a model

### auto_analyze.py

- Runs evaluation on existing trained model
- Generates confusion matrix
- Exports to ONNX
- Saves all logs to `logs/` folder
- **Does NOT retrain** - just analyzes existing model
- **Run after training** to get reports

### train_production.py (optional)

- Trains YOLOv8s model (better accuracy, 100 epochs)
- Takes 1-2 hours
- **Run only if** you want better results

## Quick Commands

### I just want to test everything works:

```bash
python train_baseline.py     # Train (wait 15-30 min)
python auto_analyze.py       # Get all reports
```

### I want to retrain:

```bash
# Delete old model
rm -rf runs/

# Train again
python train_baseline.py

# Analyze new model
python auto_analyze.py
```

### I want better accuracy:

```bash
python train_production.py   # 1-2 hours
python auto_analyze.py       # Analyze production model
```

### I just want to check GPU:

```bash
python tests/test_gpu.py
```

### I want to test API:

```bash
python api.py                # Terminal 1
python tests/test_api.py     # Terminal 2
```

## Common Questions

**Q: Do I run train_baseline.py every time?**
A: No. Once trained, the model is saved. Just run `auto_analyze.py` to regenerate reports.

**Q: What if I change the code?**
A: If you change training code → retrain. If you change evaluation code → just run `auto_analyze.py`.

**Q: Can I skip train_baseline.py?**
A: No. You need at least one trained model before running `auto_analyze.py`.

**Q: What's the difference between train_baseline and train_production?**
A: Baseline = fast (YOLOv8n, 50 epochs). Production = accurate (YOLOv8s, 100 epochs).

## Your Current Status

✅ Dataset downloaded
❌ No trained model yet

**Next step**: `python train_baseline.py`
