# Samples

Utilities for creating small, synthetic datasets that exercise the toolkit
without downloading the full production dataset.

## Quick Start

```bash
python samples/generate_sample_dataset.py
```

This generates `samples/sample_dataset/data.yaml` with train/valid/test splits.
You can then point the CLI at the generated data by setting an override config:

```bash
python samples/generate_sample_dataset.py --output samples/sample_dataset
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
python -m pcb_defect_detection.cli data-download --skip-roboflow
```

## Configuration Override

The `sample_config.yaml` file configures the toolkit to use the generated
`data.yaml`. Copy and adjust it for custom experiments:

```yaml
data:
  dataset_path_file: samples/sample_dataset/data.yaml
  fallback_patterns:
    - samples/sample_dataset/data.yaml
```

Place this alongside your run logs or version-control it to ensure reproducible
experiments.
