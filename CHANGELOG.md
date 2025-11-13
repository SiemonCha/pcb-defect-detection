# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-11-10

### Added

- Optional Weights & Biases tracking helpers with unit coverage and richer documentation.
- ONNX benchmarking command documentation and container usage guides.

### Changed

- CLI now exposes clearer help text, command descriptions, and improved dependency hints.
- Training output prints configuration summaries, tracking status, and key metrics on completion.
- Quick start, workflow, and troubleshooting docs expanded with verification steps, tracking guidance, and Docker instructions.

### Fixed

- CLI help now lists the ONNX benchmarking command and includes more descriptive epilogues.

## [0.2.0] - 2025-11-10

### Added

- Editable packaging metadata, sample dataset generator, and expanded dependency set.
- Unified namespace so both `src.*` and `pcb_defect_detection.*` imports work reliably.
- Initial automated test suite for CLI, configuration loader, confusion plotting, ONNX export, and dataset checks.
- GitHub Actions workflow that installs the project and runs pytest.

### Changed

- Centralized configuration in `pcb_defect_detection/config/default.yaml` with override support.
- Updated README with professional documentation and CLI instructions.

### Removed

- Legacy script locations in favour of the `pcb_defect_detection` package layout.

## [0.1.0] - 2025-11-05

### Added

- Baseline project structure, YOLO training utilities, and FastAPI inference server.
- Initial documentation and helper scripts for running the pipeline end-to-end.
