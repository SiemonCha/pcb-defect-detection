# PCB Defect Detection

Deep learning PCB defect detection using YOLOv8 with production-ready features.

## 🚀 Quick Start

```bash
# 1. Install dependencies
python src/install.py

# 2. Download dataset
python src/data_download.py

# 3. Train model
python scripts/train_baseline.py

# 4. Run complete analysis (auto-generates all reports)
python scripts/auto_analyze.py
```

**That's it!** All results saved to `logs/` and `outputs/` folders.

## 📁 Project Structure

```
pcb-defect-detection/
├── src/                    # Core source code
│   ├── api.py             # REST API server
│   ├── data_download.py   # Dataset downloader
│   └── install.py         # Dependency installer
│
├── scripts/               # Training & analysis scripts
│   ├── train_baseline.py        # Fast training (YOLOv8n)
│   ├── train_production.py      # Production training (YOLOv8s)
│   ├── evaluate.py              # Model evaluation
│   ├── confusion_matrix.py      # Performance analysis
│   ├── export_onnx.py           # ONNX optimization
│   ├── transfer_learning.py     # Fine-tuning
│   └── auto_analyze.py          # Run all analysis (auto-log)
│
├── tests/                 # Testing scripts
│   ├── test_gpu.py       # GPU detection test
│   ├── test_api.py       # API testing
│   └── run_tests.py      # Complete test suite
│
├── configs/              # Configuration files
│   ├── requirements/     # Platform-specific dependencies
│   ├── requirements.txt  # All dependencies
│   └── .gitignore       # Git ignore rules
│
├── data/                 # Dataset (auto-downloaded)
│   └── printed-circuit-board-2/
│       └── data.yaml
│
├── logs/                 # All logs (auto-generated)
│   ├── evaluation_log.json
│   ├── evaluation_report_*.txt
│   ├── confusion_matrix_*.png
│   ├── performance_analysis_*.txt
│   └── onnx_benchmark_*.txt
│
├── outputs/              # All outputs (auto-generated)
│   ├── models/          # Trained models (*.pt, *.onnx)
│   ├── plots/           # Training plots, confusion matrices
│   └── reports/         # Generated reports
│
└── docs/                # Documentation
    ├── README.md        # This file
    ├── QUICK_REFERENCE.md
    ├── DEPLOYMENT.md
    └── LICENSE
```

## ✨ Features

- **Auto-Logging**: All results automatically saved to organized folders
- **Multi-GPU Support**: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS)
- **Performance Analysis**: Confusion matrix with per-class insights
- **Speed Optimization**: ONNX export (3-5x faster inference)
- **REST API**: Production-ready deployment
- **Transfer Learning**: Fine-tune on new PCB types

## 📊 Workflow

### Training
```bash
python scripts/train_baseline.py
```

### Complete Analysis (Recommended)
```bash
python scripts/auto_analyze.py
```

This runs:
- Model evaluation
- Confusion matrix generation
- ONNX export with benchmarking
- Saves all results to `logs/` folder

### Individual Scripts
```bash
python scripts/evaluate.py              # Just evaluation
python scripts/confusion_matrix.py      # Just confusion matrix
python scripts/export_onnx.py           # Just ONNX export
```

### API Deployment
```bash
python src/api.py                       # Start server
python tests/test_api.py                # Test API
```

## 📈 Results Location

After running `auto_analyze.py`:

- **Logs**: `logs/` - All timestamped reports
- **Models**: `outputs/models/` - Trained weights
- **Plots**: `outputs/plots/` - Visualizations
- **Reports**: `outputs/reports/` - Summary reports

## 🎯 For Project Submission

1. Train model: `python scripts/train_baseline.py`
2. Generate all results: `python scripts/auto_analyze.py`
3. Collect files from `logs/` and `outputs/` folders
4. Include in report:
   - Confusion matrix image
   - Performance analysis report
   - ONNX benchmark results
   - Model evaluation metrics

## 🔧 Advanced

### Transfer Learning
```bash
python scripts/transfer_learning.py --data path/to/new_data.yaml
```

### Custom Model
```bash
python scripts/evaluate.py runs/train/custom/weights/best.pt
```

## 📝 Documentation

See `docs/` folder for:
- Detailed usage guide
- Deployment instructions
- API documentation

## 🧪 Testing

```bash
python tests/run_tests.py    # Complete test suite
python tests/test_gpu.py     # GPU detection
python tests/test_api.py     # API testing
```

## 📄 License

MIT License - See `docs/LICENSE`

## 🤝 Contributing

Issues and PRs welcome at GitHub repository.
