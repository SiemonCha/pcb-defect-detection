# PCB Defect Detection

Neural-network-based defect detection for printed circuit boards using YOLOv8, packaged with supporting utilities for training, evaluation, deployment, and analysis.

## Quick Start

```bash
# 1. Install dependencies and validate environment
pip install -e ".[dev]"
python install.py
python pre_flight_check.py

# 2. (Optional) Generate a tiny synthetic dataset for local experimentation
python samples/generate_sample_dataset.py

# 3. Train the baseline model (uses config/default.yaml or your override)
pcb-dd train-baseline

# 4. Generate evaluation reports and artefacts
python auto_analyze.py

# 5. Optional: run the extended analysis suite
pcb-dd quick-analysis

# Optional: execute the full pipeline in one command
python start.py
```

### Sample dataset

For quick smoke tests, generate a miniature dataset and point the toolkit at it:

```bash
python samples/generate_sample_dataset.py --output samples/sample_dataset
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
pcb-dd train-baseline
```

The generator writes a YOLO-formatted dataset to `samples/sample_dataset/` and the
`sample_config.yaml` file configures the CLI to use it.

### Configuration overrides

Project defaults live in `config/default.yaml`. Override values by supplying your own YAML and setting `PCB_CONFIG=/path/to/override.yaml` before running commands. To pin a specific dataset `data.yaml`, set `PCB_DATASET=/path/to/data.yaml`. CLI flags still take precedence.

See `docs/configuration.md` for a detailed walkthrough of configuration layering and environment variable support.

Once the project is installed editable (`pip install -e ".[dev]"`), the console script `pcb-dd` is available globally. Without installation, prefix commands with `PYTHONPATH=$(pwd)/src` so the CLI package is importable from the repository root. Outputs are written to `logs/` and `outputs/`.

## Documentation

The docs folder provides deeper guidance:

- [Quick Start](docs/quickstart.md) – extended walkthroughs for local setup and the sample dataset.
- [Configuration Guide](docs/configuration.md) – environment variables, overrides, and dataset resolution.
- [Operational Workflows](docs/workflows.md) – training, evaluation, deployment, and monitoring playbooks.
- [Troubleshooting](docs/troubleshooting.md) – common issues and their fixes.
- [Docker Guide](docker/README.md) – container images for training and the API.

Consult `docs/README.md` for the full index.

---

## Component Overview

The repository bundles utilities that focus on two main areas: exploratory analysis for research-style work, and production capabilities needed for deployment. The key command-line entry points are listed below.

### Analysis Utilities

- `pcb-dd failure-analysis`: inspect misclassified samples with visual overlays
- `pcb-dd interpretability`: generate attention maps for model predictions
- `python -m training.hyperparameter`: run Optuna-based hyperparameter searches
- `python -m training.cross_validation`: compute cross-validation metrics and confidence intervals
- `pcb-dd dataset-analysis`: summarise label distribution and potential imbalances

### Deployment Utilities

- `pcb-dd quantize`: export an INT8-quantized ONNX model for faster inference
- `pcb-dd robustness`: evaluate robustness against synthetic corruptions
- `pcb-dd monitor`: monitor inference performance and collect logs

### Quick Access

```bash
# Run the combined analysis workflow
pcb-dd quick-analysis

# Example targeted commands
pcb-dd failure-analysis        # failure case inspection
pcb-dd interpretability        # saliency and attention maps
pcb-dd dataset-analysis        # dataset statistics
pcb-dd quantize                # ONNX quantisation
pcb-dd robustness              # robustness evaluation
python -m training.cross_validation   # cross-validation experiment
python -m training.hyperparameter     # hyperparameter search
```

See [Operational Workflows](docs/workflows.md) for detailed recipes.

---

## Project Structure

```
pcb-defect-detection/
|-- src/
|   |-- analysis/
|   |-- cli.py
|   |-- config/
|   |-- data/
|   |-- deployment/
|   |-- evaluation/
|   |-- setup/
|   `-- training/
|
|-- auto_analyze.py
|-- start.py
|-- pre_flight_check.py
|-- install.py
|-- run_tests.py
|-- tests/
|-- logs/
|-- runs/
|-- outputs/
|-- datasets/
`-- requirements.txt
```

---

## Features

### Core Capabilities

- Structured logging for runs, reports, and exported artefacts
- Multi-accelerator support: CUDA, ROCm, and Apple Metal Performance Shaders
- Evaluation utilities including confusion matrices and summary metrics
- ONNX export for accelerated inference
- FastAPI-based inference service
- Transfer learning workflow for new PCB datasets

### Extended Capabilities

- Failure analysis visualisations to inspect false positives/negatives
- Interpretability tooling (Grad-CAM and confidence curves)
- Automated hyperparameter search with Optuna
- Cross-validation driver for statistical evaluation
- Dataset-level imbalance diagnostics
- INT8 quantisation for deployment targets
- Robustness benchmarking against predefined corruptions

---

## Workflow

### Basic Workflow (Existing)

```bash
# Training
pcb-dd train-baseline

# Complete Analysis
python auto_analyze.py

# Individual Commands
pcb-dd evaluate                     # Just evaluation
pcb-dd confusion                    # Just confusion matrix
pcb-dd export-onnx                  # Just ONNX export

# API Deployment
pcb-dd api                          # Start server
pytest tests/test_api.py                                        # Test API
```

### Advanced Workflow (NEW)

#### For Academic/Research (1-2 hours)

```bash
# 1. Quick comprehensive analysis
pcb-dd quick-analysis

# 2. Deep statistical analysis (optional)
python -m training.cross_validation --quick        # 2-3 hours
python -m training.hyperparameter --quick         # 5-10 hours
```

**What you get:**

- Class distribution analysis
- Failure case visualizations
- Attention maps
- Cross-validated metrics with confidence intervals
- Optimized hyperparameters

#### For Production Deployment (30 min)

```bash
# 1. Optimize for production
pcb-dd quantize --model outputs/models/best.pt

# 2. Test robustness
pcb-dd robustness --model outputs/models/best_int8.onnx

# 3. Deploy with monitoring
pcb-dd api --model outputs/models/best_int8.onnx
pcb-dd monitor            # Track performance
```

**What you get:**

- Production-ready INT8 model (10-12ms inference)
- Robustness scores for 11 corruption types
- Real-time monitoring dashboard

---

## Results Location

After running analyses:

### Basic Results (Existing)

- **Logs**: `logs/` - All timestamped reports
- **Models**: `outputs/models/` - Trained weights
- **Plots**: `outputs/plots/` - Visualizations
- **Reports**: `outputs/reports/` - Summary reports

### Advanced Results (NEW)

- **Dataset Analysis**: `logs/dataset_analysis/` - Class distributions
- **Failure Cases**: `logs/failure_analysis/` - What model gets wrong
- **Interpretability**: `logs/interpretability/` - Attention maps
- **Robustness**: `logs/robustness/` - Edge case testing
- **Monitoring**: `logs/monitoring/` - Production metrics
- **Cross-Validation**: `logs/cross_validation_*/` - Statistical results

---

## For Project Submission

### Academic/Research Submission

1. **Train model:**

   ```bash
   pcb-dd train-baseline
   ```

2. **Generate comprehensive analysis:**

   ```bash
   pcb-dd quick-analysis
   ```

3. **Optional - Deep analysis:**

   ```bash
   python -m training.cross_validation --quick
   ```

4. **Collect results from:**

   - `logs/dataset_analysis/` -> Class distribution summary
   - `logs/failure_analysis/` -> Notable failure cases
   - `logs/interpretability/` -> Attention map visualisations
   - `logs/cross_validation_*/` -> Cross-validation metrics

5. **Include in report:**
   - Dataset analysis showing class distribution
   - Cross-validated metrics: "mAP@0.5 = 94.8% +/- 2.1%"
   - Failure case examples with explanations
   - Attention maps proving model focuses on defects
   - Confusion matrix with per-class analysis

### Production Deployment

1. **Train and optimize:**

   ```bash
   pcb-dd train-baseline
   pcb-dd quantize --model outputs/models/best.pt
   pcb-dd robustness --model outputs/models/best_int8.onnx
   ```

2. **Deploy:**

   ```bash
   pcb-dd api --model outputs/models/best_int8.onnx
   ```

3. **Monitor:**

   ```bash
   pcb-dd monitor --api-url http://localhost:8000
   ```

4. **Verify service-level targets:**
   - Measure latency (target P95 < 100 ms; reference runs show ~10-12 ms after optimisation)
   - Confirm accuracy on validation data (target > 85%; baseline reference ~99.5%)
   - Calculate error rate (target < 1%)
   - Review robustness results across corruption scenarios (target: stable on at least 8 of 11 synthetic tests)

---

## Advanced

### Transfer Learning

```bash
pcb-dd transfer-learning --data path/to/new_data.yaml
```

### Custom Model

```bash
pcb-dd evaluate runs/train/custom/weights/best.pt
```

### Hyperparameter Optimization

```bash
# Quick search (10 trials, ~5 hours)
python -m training.hyperparameter --quick

# Full search (20 trials, ~15-30 hours)
python -m training.hyperparameter --trials 20
```

The best parameters are written to `logs/best_hyperparameters_*.yaml`; apply them to your training configuration manually.

### Cross-Validation

```bash
# Quick 3-fold CV (~2-3 hours)
python -m training.cross_validation --quick

# Full 5-fold CV (~6-10 hours)
python -m training.cross_validation --folds 5

# Thorough 10-fold CV (~12-20 hours)
python -m training.cross_validation --folds 10
```

---

## Performance Metrics

Typical reference results (baseline YOLOv8n trained on the supplied dataset):

- mAP@0.5: approximately 99.5% (single hold-out split)
- Cross-validated mAP@0.5: approximately 94.8% +/- 2.1% (95% confidence interval)
- ONNX (FP16) latency: roughly 36 ms per image on an RTX-class GPU
- ONNX INT8 latency: roughly 10-12 ms per image after quantisation
- Robustness evaluation: reliable on 8 of 11 common corruption scenarios

When preparing reports or documentation, reference the exact figures obtained in your environment and include supporting artefacts (confusion matrices, robustness tables, quantisation benchmarks, etc.).

---

## Testing

```bash
python tests/run_tests.py    # Complete test suite
python tests/test_gpu.py     # GPU detection
pytest tests/test_api.py     # API testing

# Additional checks
pcb-dd verify-setup        # Verify dependencies
pcb-dd robustness          # Edge case testing
pcb-dd monitor             # Production monitoring
```

---

## Contributing & Support

We welcome contributions. Please review [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md) before opening a pull request. For questions or
bug reports, open an issue with logs from `logs/` and the commands run.

### Experiment tracking

Enable Weights & Biases logging by updating `config/default.yaml` (or an override):

```yaml
tracking:
  enabled: true
  project: pcb-defect-detection
  entity: your-team
```

Then run `wandb login` once and launch training (`pcb-dd train-baseline`).

### Docker

Build container images via `docker compose build` (see `docker/README.md`) to run the API
or training jobs in a reproducible environment.
