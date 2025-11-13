# Quick Start

This guide walks through a fresh clone to a working training run and evaluation. The
process assumes macOS or Linux with Python 3.10+.

## 1. Clone and environment setup

```bash
git clone https://github.com/siemoncha/pcb-defect-detection.git
cd pcb-defect-detection
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> Tip: quote `".[dev]"` so shells like zsh do not expand the brackets.

Confirm the CLI resolves:
```bash
pcb-dd --help
```

## 2. Validate prerequisites

```bash
python install.py
python pre_flight_check.py
```

These scripts confirm Python dependencies, optional accelerators, and filesystem
permissions. Resolve reported issues before continuing.

## 3. Option A – work with the sample dataset

Use the synthetic dataset to validate workflows without large downloads:

```bash
python samples/generate_sample_dataset.py
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
```

The override points the toolkit at `samples/sample_dataset/data.yaml` and tightens
training parameters for fast iterations. Verify the dataset entry resolves:
```bash
pcb-dd dataset-analysis
```

## 4. Option B – download the reference dataset

```bash
pcb-dd data-download
```

This command uses the configuration defaults to locate or fetch the production dataset.
Set `PCB_DATASET=/absolute/path/to/data.yaml` to pin a custom location and avoid further
glob scanning.

## 5. Train a baseline model

```bash
pcb-dd train-baseline --epochs 10 --batch 8
```

The trainer prints a configuration summary, tracking status, and periodic progress
updates. Logs stream to `logs/training/` and checkpoints to `runs/train/baseline_yolov8n*`.
Adjust arguments as needed; all CLI flags mirror Ultralytics parameters.

## 6. Evaluate and analyse

```bash
pcb-dd evaluate
pcb-dd confusion --split test
pcb-dd quick-analysis --skip-robustness
```

Artifacts land under `logs/` and `runs/*/analysis/`. Review confusion matrices,
per-class metrics, and detailed reports for regression detection.

## 7. Deploy and monitor (optional)

```bash
pcb-dd export-onnx
pcb-dd quantize --model runs/train/baseline_yolov8n*/weights/best.pt
pcb-dd api --model runs/train/baseline_yolov8n*/weights/best.pt
pcb-dd monitor --duration 120
```

Run the monitor in a separate terminal while the API is active to produce latency and
accuracy summaries.

## 8. Enable experiment tracking (optional)

Edit your override config or environment to set `tracking.enabled: true`, then log in to
Weights & Biases (`wandb login`). The training commands automatically initialise runs
with the configured project/entity.

## 9. Next steps

- Explore the [Operational Workflows](workflows.md) guide for production-ready
  pipelines, robustness testing, and hyperparameter tuning.
- Review the [Configuration Guide](configuration.md) to understand overrides and
  environment variables.
- Visit the [Troubleshooting](troubleshooting.md) page for common errors and fixes.
