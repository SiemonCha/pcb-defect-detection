# PCB Defect Detection

Automated PCB defect detection using YOLOv8. Train models, run evaluations, deploy to production — everything you need for computer vision in manufacturing.

Built for a university ML project, but it actually works. Runs on NVIDIA/AMD GPUs, Apple Silicon, and CPU-only systems.

## Why this exists

Most PCB detection tutorials stop at "congrats, you trained a model!" This project goes further:

- **Cross-platform** - Works on whatever hardware you have (tested on NVIDIA, Apple M1, CPU)
- **Actually deploys** - REST API, Docker containers, performance monitoring
- **Production features** - ONNX optimization, INT8 quantization, robustness testing
- **Research-friendly** - Cross-validation, hyperparameter tuning, statistical analysis
- **Honest logging** - Everything auto-logs to `logs/` so you can write reports

I needed something that would work for both my coursework AND potentially in real manufacturing. So here we are.

## Quick Start

**Fastest path (5 minutes):**

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Generate a tiny test dataset
python samples/generate_sample_dataset.py

# 3. Train (takes ~2 minutes on GPU, ~10 on CPU)
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
pcb-dd train-baseline --epochs 10

# 4. See results
ls logs/
```

**Full workflow (30-45 minutes):**

```bash
# Runs everything: download dataset, train, evaluate, generate reports
python start.py
```

**Just want to test the API?**

```bash
# Use a pretrained model or train one first
pcb-dd train-baseline
pcb-dd api

# In another terminal
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/pcb_image.jpg"
```

## What's in the box

### Core features (tested, reliable)

- **Training** - YOLOv8n baseline (fast) or YOLOv8s production (accurate)
- **Evaluation** - Confusion matrices, per-class metrics, auto-generated reports
- **Deployment** - FastAPI server, ONNX export, Docker containers
- **Multi-platform** - Auto-detects your hardware (CUDA, ROCm, MPS, CPU)

### Advanced features (work but may need tweaking)

- **Failure analysis** - Visualize what the model gets wrong
- **Interpretability** - Attention maps showing where the model looks
- **Cross-validation** - Statistical confidence intervals (takes hours)
- **Hyperparameter tuning** - Optuna-based search (also takes hours)
- **Robustness testing** - How well does it handle noise, blur, etc.
- **INT8 quantization** - 3-4x speedup for deployment

The advanced stuff is there because I wanted to learn it. Your mileage may vary.

## Project Structure

```
pcb-defect-detection/
├── src/                      # Main codebase
│   ├── training/             # Baseline, production, transfer learning
│   ├── evaluation/           # Metrics, confusion, interpretability
│   ├── deployment/           # API, ONNX export, quantization
│   ├── analysis/             # Dataset stats, failure cases
│   └── cli.py                # Command-line interface
│
├── tests/                    # Unit tests (basic coverage)
├── samples/                  # Sample dataset generator
├── docker/                   # Docker configs
├── docs/                     # Detailed guides
│
├── start.py                  # One-command full workflow
├── auto_analyze.py           # Generate all reports
└── requirements.txt          # Dependencies
```

After training, check these directories:

- `runs/train/` - Model checkpoints and training curves
- `logs/` - All reports, metrics, visualizations
- `outputs/` - Final models and deployment artifacts

## Installation

**Recommended (virtual environment):**

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install project
pip install -e ".[dev]"

# Platform-specific packages (PyTorch, ONNX Runtime)
python install.py
```

The `install.py` script auto-detects your GPU and installs the right PyTorch/ONNX builds. If it fails, check `docs/troubleshooting.md`.

**Requirements:**

- Python 3.10 or 3.11 (3.12 works but 3.11 recommended)
- 8GB+ RAM
- GPU recommended but not required

## Common Workflows

### Training a model

```bash
# Fast baseline (good enough for most cases)
pcb-dd train-baseline --epochs 50

# Higher accuracy production model
pcb-dd train-production --epochs 100

# Fine-tune on your own data
pcb-dd transfer-learning --data path/to/your/data.yaml
```

Training auto-logs everything to `logs/training/`.

### Evaluating performance

```bash
# Basic metrics
pcb-dd evaluate

# Confusion matrix with per-class breakdown
pcb-dd confusion

# Run everything (dataset analysis, failures, interpretability)
pcb-dd quick-analysis
```

Check `logs/` for generated reports and visualizations.

### Deploying to production

```bash
# Export to ONNX (faster inference)
pcb-dd export-onnx

# Quantize to INT8 (even faster)
pcb-dd quantize

# Start REST API
pcb-dd api --model runs/train/baseline_yolov8n*/weights/best.pt

# Test it
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "conf_threshold=0.25"
```

API docs at `http://localhost:8000/docs`

### Research/analysis features

**Cross-validation (statistical confidence):**

```bash
# Quick 3-fold CV (~2 hours)
python -m training.cross_validation --quick

# Full 5-fold CV (~6 hours)
python -m training.cross_validation --folds 5
```

Gives you proper error bars: "mAP@0.5 = 94.8% ± 2.1%"

**Hyperparameter tuning:**

```bash
# Quick search (10 trials, ~5 hours)
python -m training.hyperparameter --quick

# Thorough search (50 trials, ~2 days)
python -m training.hyperparameter --trials 50
```

Results saved to `logs/best_hyperparameters_*.yaml`

**Robustness testing:**

```bash
# Test against noise, blur, occlusions, etc.
pcb-dd robustness --samples 20
```

**Failure analysis:**

```bash
# See what the model gets wrong
pcb-dd failure-analysis --top 30
```

## Performance

Typical results on the included PCB dataset (your results will vary):

| Model                | mAP@0.5 | Inference (ONNX) | Size |
| -------------------- | ------- | ---------------- | ---- |
| YOLOv8n (baseline)   | ~99.5%  | ~36ms            | 6MB  |
| YOLOv8n (INT8)       | ~99.3%  | ~12ms            | 3MB  |
| YOLOv8s (production) | ~99.7%  | ~48ms            | 22MB |

Tested on:

- NVIDIA RTX 3080 (primary testing)
- Apple M1 Pro (works, slower)
- CPU-only (works, very slow)
- AMD ROCm (untested but should work)

## Docker

```bash
# Build images
cd docker
docker compose build

# Run training
docker compose run --rm trainer pcb-dd train-baseline

# Run API
docker compose up api
```

See `docker/README.md` for GPU setup.

## Configuration

Project config lives in `src/pcb_defect_detection/config/default.yaml`. Override it:

```bash
# Option 1: Environment variable
export PCB_CONFIG=/path/to/my_config.yaml

# Option 2: Specify dataset directly
export PCB_DATASET=/path/to/data.yaml
```

See `docs/configuration.md` for details.

## For Academic Submissions

**What to include in your report:**

```bash
# 1. Train and analyze
python start.py

# 2. Run comprehensive analysis
pcb-dd quick-analysis

# 3. Optional: statistical analysis
python -m training.cross_validation --quick
```

**Then collect:**

- All files from `logs/` directory
- Confusion matrices from `logs/confusion_matrix_*.png`
- Training curves from `runs/train/*/results.png`
- Model metrics from `logs/evaluation_*.txt`

**Report structure suggestion:**

1. Dataset analysis (class distribution, samples per class)
2. Training approach (architecture choice, hyperparameters)
3. Results (mAP, precision, recall with confidence intervals if you ran CV)
4. Failure analysis (what the model struggles with)
5. Deployment considerations (ONNX speedup, production requirements)

## Troubleshooting

**Installation issues:**

```bash
# If install.py fails
pip install --upgrade pip
pip cache purge
python install.py --force-cpu  # Try CPU-only first

# Check what's wrong
python pre_flight_check.py
```

**Training crashes:**

```bash
# Reduce batch size
pcb-dd train-baseline --batch 4

# Disable mixed precision (for AMD/MPS)
# Edit training script, set amp=False
```

**Import errors:**

```bash
# Make sure you installed editable
pip install -e ".[dev]"

# Check installation
pcb-dd --help
```

**Dataset not found:**

```bash
# Generate sample dataset
python samples/generate_sample_dataset.py
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml

# Or set your dataset path
export PCB_DATASET=/path/to/your/data.yaml
```

More help: `docs/troubleshooting.md`

## Known Issues

- **AMD ROCm support** - Theoretically works but untested. You'll probably need to tweak AMP settings.
- **Cross-validation** - Takes hours. Start with `--quick` mode.
- **Hyperparameter tuning** - Can take days. Use `--quick` for testing.
- **W&B tracking** - Disabled by default. Enable in config if you want it.

## Contributing

This started as coursework but contributions welcome:

1. Fork it
2. Create a branch (`git checkout -b feature/whatever`)
3. Make your changes
4. Add tests if possible
5. Submit a PR

See `CONTRIBUTING.md` for coding standards.

## License

MIT License - see `LICENSE` file.

Free to use for academic or commercial projects. Attribution appreciated but not required.

## Acknowledgments

- **YOLOv8** by Ultralytics - the actual detection engine
- **Roboflow** - dataset hosting and augmentation
- **PCB Defects Dataset** - various contributors on Roboflow Universe
- My university ML course for forcing me to actually finish this

## Citation

If you use this for research:

```bibtex
@misc{pcb-defect-detection,
  author = {Sansiri Charoenpong},
  title = {PCB Defect Detection: End-to-end YOLOv8 Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/siemoncha/pcb-defect-detection}
}
```

## Contact

- GitHub Issues for bugs/questions
- Pull requests for contributions
- Or just fork it and make it your own

Built with frustration, coffee, and surprisingly few Stack Overflow tabs.

---

**Status:** Working and tested for coursework. Production use at your own risk (but it should be fine).

**Last updated:** November 2025
