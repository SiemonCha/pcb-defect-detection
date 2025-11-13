# Configuration Guide

This project centralises settings in `pcb_defect_detection/config/default.yaml` and
allows you to override them per-environment. The key concepts:

## Default Settings

- `training.*`: baseline and production training defaults.
- `data.dataset_path_file`: location of a text file used to cache the resolved
  dataset `data.yaml` path.
- `data.fallback_patterns`: glob patterns probed when the dataset cache is
  empty.

## Override Precedence

1. `PCB_DATASET` environment variable – explicit path to a `data.yaml` file.
2. `PCB_CONFIG` – YAML override merged with the defaults.
3. Cached value in `dataset_path_file` (written automatically once located).
4. Glob patterns under `data.fallback_patterns`.

## Common Workflows

### Use the bundled sample dataset

```bash
python samples/generate_sample_dataset.py
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
pcb-dd train-baseline
```

### Validate against a downloaded dataset

```bash
pcb-dd data-download
export PCB_DATASET=$(pwd)/data/YourDataset/data.yaml
pcb-dd evaluate
```

### Custom per-user overrides

Create `~/.config/pcb-defect-detection/config.yaml`:

```yaml
training:
  baseline:
    epochs: 30
    batch: 4
```

Then set:

```bash
export PCB_CONFIG=~/.config/pcb-defect-detection/config.yaml
```

## Tips

- The helper `data.resolve_dataset_yaml()` writes the resolved path back to the
  cache file so future runs launch instantly.
- Keep project-specific overrides under version control (e.g. inside
  `config/overrides/production.yaml`).
- For CI pipelines, set `PCB_DATASET` to a CI artifact so tests can resolve the
  dataset without guessing through glob patterns.

## Tracking

The `tracking` section controls optional experiment logging (currently Weights & Biases):

```yaml
tracking:
  enabled: false
  project: pcb-defect-detection
  entity: your-team
  run_name_prefix: pcbdd
  tags:
    - baseline
```

- Set `enabled: true` to allow the training scripts to initialise a W&B run.
- Configure `project`, `entity`, and optional tags.
- Install `wandb` (already listed in `requirements.txt`) and authenticate via
  `wandb login` before enabling.
