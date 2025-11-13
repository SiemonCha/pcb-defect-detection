# Contributing Guide

Thank you for considering a contribution. This project values clear code, reproducible
experiments, and thorough documentation. The sections below describe expectations for
pull requests.

## Before you start

- Fork the repository and create a feature branch (`feat/...`, `fix/...`, or `docs/...`).
- Install development dependencies:
  ```bash
  pip install -e ".[dev]"
  ```
- Run the quick smoke tests to ensure a clean baseline:
  ```bash
  pytest
  ```

## Coding standards

- Use the src/ layout and keep new modules inside the `pcb_defect_detection` package.
- Prefer configuration via `config/default.yaml` plus overrides instead of hard-coded
  constants.
- Follow the logging style already present in the training scripts (concise, timestamped
  summaries every ~15 seconds).
- Type hints are encouraged for public functions.

## Workflow for changes

1. Implement the change with accompanying unit tests where practical. New functionality
   should include tests under `tests/`.
2. Run the formatter/linter:
   ```bash
   ruff check src tests samples
   ```
3. Execute the tests:
   ```bash
   pytest
   ```
4. Update documentation if behaviour or CLI usage changes (`README.md`, `docs/`, or inline
   docstrings).
5. Update `CHANGELOG.md` under the *Unreleased* section (add one if necessary).
6. Submit a pull request describing the motivation, approach, and testing performed.

## Commit guidelines

- Keep commits focused and well described (`feat:`, `fix:`, `docs:` prefixes help).
- Rebase onto `main` before opening the PR to avoid merge commits.

## Code review

- PRs require one approval. Reviews focus on correctness, maintainability, and impact on
  training reproducibility.
- Be prepared to provide benchmark results or logs for performance sensitive changes.

## Reporting issues

Include:
- Command(s) executed
- Environment details (OS, Python version, GPU info)
- Full traceback or log snippet

Use the issue templates (or create a detailed report) so we can reproduce the problem
quickly.
