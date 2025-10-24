# PCB Defect Detection

Deep learning PCB defect detection using YOLOv8 with multi-platform GPU support.

## Features

- **Multi-GPU Support**: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS)
- **Automated Setup**: Platform detection and dependency installation
- **YOLOv8 Detection**: Baseline (n) and Production (s) models
- **Evaluation Metrics**: mAP, precision, recall, per-class AP

## Requirements

- Python 3.8-3.12
- Git

### GPU Requirements

| Platform      | Requirements                  |
| ------------- | ----------------------------- |
| NVIDIA        | CUDA 11.8+, drivers installed |
| AMD           | ROCm 6.0+, Linux only         |
| Apple Silicon | macOS 12.0+, M1/M2            |

## Installation

```bash
git clone https://github.com/SiemonCha/pcb-defect-detection.git
cd pcb-defect-detection

# Create environment
conda create -n pcb312 python=3.12
conda activate pcb312

# Auto-install dependencies
python install.py
```

## Quick Start

```bash
# 1. Download dataset
python data_download.py

# 2. Train baseline (fast)
python train_baseline.py

# 3. Evaluate
python evaluate.py

# 4. Train production (better accuracy)
python train_production.py
```

## Project Structure

```
pcb-defect-detection/
├── data/
│   └── printed-circuit-board-2/
│       └── data.yaml              # Dataset config
├── requirements/
│   ├── requirements-common.txt    # Shared dependencies
│   ├── requirements-cuda.txt      # NVIDIA GPU
│   ├── requirements-rocm.txt      # AMD GPU
│   └── requirements-mac.txt       # Apple Silicon
├── train_baseline.py              # YOLOv8n (fast)
├── train_production.py            # YOLOv8s (accurate)
├── evaluate.py                    # Test set evaluation
├── data_download.py               # Roboflow dataset
└── install.py                     # Auto-installer
```

## Training Details

### Baseline (YOLOv8n)

- **Speed**: ~50 epochs in 15-30 min (GPU)
- **Size**: 6.3M params
- **Target**: Quick validation, >80% mAP@0.5

### Production (YOLOv8s)

- **Speed**: ~100 epochs in 1-2 hours (GPU)
- **Size**: 11.2M params
- **Target**: >85% mAP@0.5, <100ms inference

## Platform Support

### AMD ROCm

**Important**: PyTorch ROCm uses `'cuda'` as device string (not `'xpu'`). Detection via `torch.version.hip`.

**Check GPU**:

```bash
rocm-smi
python test_gpu.py
```

### NVIDIA CUDA

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Troubleshooting

### AMD GPU not detected

```bash
# Check ROCm
rocm-smi

# Check PyTorch
python -c "import torch; print(torch.version.hip)"

# User permissions
sudo usermod -aG video $USER
```

### NVIDIA GPU not detected

```bash
nvidia-smi
nvcc --version  # Check CUDA toolkit
```

### Apple Silicon issues

- Use native ARM64 Python (not Rosetta)
- Install XCode tools: `xcode-select --install`

## Common Errors

| Error                 | Fix                          |
| --------------------- | ---------------------------- |
| `data.yaml not found` | Run `data_download.py` first |
| `best.pt not found`   | Train model before evaluate  |
| ROCm not using GPU    | Check user in `video` group  |
| CUDA out of memory    | Reduce `batch` size          |

## License

MIT License - See [LICENSE](LICENSE)

## Contributing

1. Fork repo
2. Create feature branch
3. Test on your platform
4. Submit PR with platform tested

## Dataset

Uses Roboflow 100 PCB dataset. See `data_download.py` for source.
