# PCB Defect Detection

Neural-network-based defect detection for printed circuit boards using YOLOv8, packaged with supporting utilities for training, evaluation, deployment, and analysis.

## Quick Start

```bash
# 1. Install dependencies and validate environment
python install.py
python pre_flight_check.py

# 2. Download the reference dataset
python -m cli data-download

# 3. Train the baseline model
python -m cli train-baseline

# 4. Generate evaluation reports and artefacts
python auto_analyze.py

# 5. Optional: run the extended analysis suite
python -m cli quick-analysis

# Optional: execute the full pipeline in one command
python start.py
```

Set `PYTHONPATH=$(pwd)/src` (or prefix commands with `PYTHONPATH=src`) so the CLI package is importable from the repository root. Outputs are written to `logs/` and `outputs/`.

---

## Component Overview

The repository bundles utilities that focus on two main areas: exploratory analysis for research-style work, and production capabilities needed for deployment. The key command-line entry points are listed below.

### Analysis Utilities

- `python -m cli failure-analysis`: inspect misclassified samples with visual overlays
- `python -m cli interpretability`: generate attention maps for model predictions
- `python -m training.hyperparameter`: run Optuna-based hyperparameter searches
- `python -m training.cross_validation`: compute cross-validation metrics and confidence intervals
- `python -m cli dataset-analysis`: summarise label distribution and potential imbalances

### Deployment Utilities

- `python -m cli quantize`: export an INT8-quantized ONNX model for faster inference
- `python -m cli robustness`: evaluate robustness against synthetic corruptions
- `python -m cli monitor`: monitor inference performance and collect logs

### Quick Access

```bash
# Run the combined analysis workflow
python -m cli quick-analysis

# Example targeted commands
python -m cli failure-analysis        # failure case inspection
python -m cli interpretability        # saliency and attention maps
python -m cli dataset-analysis        # dataset statistics
python -m cli quantize                # ONNX quantisation
python -m cli robustness              # robustness evaluation
python -m training.cross_validation   # cross-validation experiment
python -m training.hyperparameter     # hyperparameter search
```

See [ADVANCED_COMPONENTS_GUIDE.md](docs/ADVANCED_COMPONENTS_GUIDE.md) for details.

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
python -m cli train-baseline

# Complete Analysis
python auto_analyze.py

# Individual Commands
python -m cli evaluate                     # Just evaluation
python -m cli confusion                    # Just confusion matrix
python -m cli export-onnx                  # Just ONNX export

# API Deployment
python -m cli api                          # Start server
python tests/test_api.py                                        # Test API
```

### Advanced Workflow (NEW)

#### For Academic/Research (1-2 hours)

```bash
# 1. Quick comprehensive analysis
python -m cli quick-analysis

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
python -m cli quantize --model outputs/models/best.pt

# 2. Test robustness
python -m cli robustness --model outputs/models/best_int8.onnx

# 3. Deploy with monitoring
python -m cli api --model outputs/models/best_int8.onnx
python -m cli monitor            # Track performance
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
   python -m cli train-baseline
   ```

2. **Generate comprehensive analysis:**

   ```bash
   python -m cli quick-analysis
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
   python -m cli train-baseline
   python -m cli quantize --model outputs/models/best.pt
   python -m cli robustness --model outputs/models/best_int8.onnx
   ```

2. **Deploy:**

   ```bash
   python -m cli api --model outputs/models/best_int8.onnx
   ```

3. **Monitor:**

   ```bash
   python -m cli monitor --api-url http://localhost:8000
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
python -m cli transfer-learning --data path/to/new_data.yaml
```

### Custom Model

```bash
python -m cli evaluate runs/train/custom/weights/best.pt
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
python tests/test_api.py     # API testing

# Additional checks
python -m cli verify-setup        # Verify dependencies
python -m cli robustness          # Edge case testing
python -m cli monitor             # Production monitoring
```

---

## Documentation

See `docs/` folder for:

- **QUICK_REFERENCE.md** - Command reference
- **DEPLOYMENT.md** - Deployment guide
- **ADVANCED_COMPONENTS_GUIDE.md** - Detailed note on analysis and production tooling
- **SUMMARY_OF_ADDITIONS.md** - Overview of supplemental functionality
- **INSTALLATION_GUIDE.md** - Step-by-step setup instructions

---

## Installation

### Basic Installation (Existing)

```bash
# 1. Clone repository
git clone <your-repo>
cd pcb-defect-detection

# 2. Install dependencies
python install.py

# 3. Verify installation
python tests/test_gpu.py
```

### Install Advanced Components (NEW)

```bash
# Install additional dependencies for advanced features
pip install optuna scipy

# Verify advanced components
python -m cli verify-setup

# Run quick analysis to test
python -m cli quick-analysis
```
