# PCB Defect Detection

This repository contains a deep learning-based solution for detecting defects in Printed Circuit Boards (PCBs) using YOLOv8.

## Features

- Multi-platform support (Linux, macOS, Windows)
- GPU acceleration support for:
  - NVIDIA GPUs (CUDA)
  - AMD GPUs (ROCm)
  - Apple Silicon (M1/M2)
- Automated environment setup
- YOLOv8-based object detection
- Training and evaluation scripts
- Production-ready inference

## System Requirements

### Common Requirements

- Python 3.8 - 3.12
- Git

### GPU Support Requirements

#### NVIDIA GPUs

- NVIDIA GPU with CUDA capability
- NVIDIA drivers and CUDA toolkit 11.8+

#### AMD GPUs

- AMD GPU with ROCm support
- ROCm 6.0+ installed
- Linux operating system

#### Apple Silicon

- M1/M2 Mac
- macOS 12.0+

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SiemonCha/pcb-defect-detection.git
cd pcb-defect-detection
```

2. Create a Conda environment (recommended):

```bash
conda create -n pcb312 python=3.12
conda activate pcb312
```

3. Install dependencies:

```bash
python install.py
```

The installation script will automatically detect your platform and install the appropriate versions of PyTorch and other dependencies.

## Project Structure

```
pcb-defect-detection/
├── data/                    # Dataset directory
│   └── data.yaml           # Dataset configuration
├── requirements/            # Platform-specific requirements
│   ├── requirements-common.txt
│   ├── requirements-cuda.txt
│   ├── requirements-rocm.txt
│   └── requirements-mac.txt
├── train_baseline.py       # Training script
├── train_production.py     # Production training script
├── evaluate.py             # Evaluation script
├── data_download.py        # Dataset download script
└── install.py             # Automated installation script
```

## Training

1. Prepare your dataset:

   - Organize your PCB images and annotations
   - Update `data/data.yaml` with your dataset paths

2. Start training:

```bash
python train_baseline.py
```

The script will automatically detect and use the best available hardware:

- NVIDIA GPU with CUDA
- AMD GPU with ROCm
- Apple Silicon GPU
- CPU (if no GPU is available)

## Model Training Parameters

The default training configuration includes:

- Base model: YOLOv8n
- Input size: 640x640
- Batch size: 8
- Epochs: 50
- Learning rate: Auto-configured
- Data augmentation: Enabled
- Early stopping: Enabled (patience=10)

## Hardware Support Details

### NVIDIA GPUs

- Uses CUDA acceleration
- Supported by PyTorch with CUDA (cu118)
- Recommended for Windows and Linux

### AMD GPUs

- Uses ROCm acceleration
- Supported by PyTorch with ROCm 6.0
- Linux only
- Compatible with recent AMD GPUs (RX 6000 and 7000 series)

### Apple Silicon

- Uses Metal Performance Shaders (MPS)
- Native ARM64 support
- Optimized for M1/M2 chips

## Troubleshooting

### Common Issues

1. GPU not detected:

   - NVIDIA: Check `nvidia-smi`
   - AMD: Check `rocm-smi`
   - Apple: Ensure using macOS 12.0+

2. Installation fails:
   - Check Python version compatibility
   - Verify GPU drivers are installed
   - Ensure ROCm is properly installed (AMD)

### Platform-Specific Notes

#### AMD GPU Users

- Ensure ROCm is installed and working
- Check GPU compatibility with ROCm
- Verify user is in video group

#### NVIDIA GPU Users

- Verify CUDA toolkit installation
- Check NVIDIA driver version
- Ensure sufficient GPU memory

#### Apple Silicon Users

- Use native ARM64 Python
- Ensure XCode tools are installed

## License

See the [LICENSE](LICENSE) file for license rights and limitations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
