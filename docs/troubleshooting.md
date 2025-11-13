# Troubleshooting

This page lists common issues, error messages, and recommended fixes.

## Installation issues

### `zsh: no matches found: .[dev]`
Quote the extras specifier:
```bash
pip install -e ".[dev]"
```

### `ModuleNotFoundError: No module named 'pcb_defect_detection'`
Ensure the editable install succeeded and the virtual environment is active. Run:
```bash
pip install -e ".[dev]"
which pcb-dd
```
The CLI should resolve to the project environment.

## Dataset resolution

### `Unable to locate a dataset data.yaml`
- Set `PCB_DATASET=/absolute/path/to/data.yaml`, or
- Generate the sample dataset via `python samples/generate_sample_dataset.py` then export
  `PCB_CONFIG=$(pwd)/samples/sample_config.yaml`.

Clear the cache file if it references stale paths:
```bash
rm dataset_path.txt
```

## Training problems

### Training appears to hang with no progress
Logs are throttled to fire every ~10 seconds. Set `LOG_INTERVAL=5` (environment
variable) to increase verbosity:
```bash
LOG_INTERVAL=5 pcb-dd train-baseline
```

### `CUDA out of memory`
- Lower the batch size: `pcb-dd train-baseline --batch 4`
- Reduce image size: `--imgsz 512`
- Use gradient accumulation: `--accumulate 2`

## Evaluation and ONNX export

### `onnxruntime` import errors
Install the export extras:
```bash
pip install onnxruntime onnxslim
```

### INT8 export fails
The quantization step gracefully falls back to FP32 ONNX. Verify:
```bash
pcb-dd export-onnx --model path/to/best.pt --format onnx
```
Then run `pcb-dd quantize` on the generated file.

## API / Monitoring

### API returns 503 `Model not loaded`
Ensure a trained model exists under `runs/train/.../weights/best.pt`. Restart the server
with the `--model` flag pointing to the checkpoint.

### Monitor reports `No test images found`
Generate or download a dataset so `test/images` exists. For the sample dataset:
```bash
python samples/generate_sample_dataset.py
export PCB_CONFIG=$(pwd)/samples/sample_config.yaml
```

## Diagnostics checklist

- `pcb-dd verify-setup` – dependency and accelerator checks.
- `pcb-dd dataset-analysis` – confirms dataset integrity.
- `python run_tests.py` – smoke test suite runner.

If a problem persists, capture logs from `logs/` and open an issue with the exact
command, environment details, and traceback.

## CLI messages

### `[ERROR] Missing dependency '...'`
Install the referenced package (see `requirements.txt`) or disable the command requiring
it. For example, run `pip install onnxruntime` before using `pcb-dd benchmark-onnx`.

### `Unknown command`
Run `pcb-dd --help` to review available sub-commands and syntax.

## Tracking

### W&B run not initialised
Ensure `tracking.enabled` is set to `true` in your config override, the `wandb` package is
installed, and you have authenticated via `wandb login`.

### Disable tracking for offline experimentation
Set `tracking.enabled: false` or export `WANDB_MODE=disabled` to silence tracking logs.
```
