from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


def configure_wandb(
    tracking_cfg: Dict[str, Any],
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
):
    """Initialise a Weights & Biases run if tracking is enabled.

    Returns the run object or ``None`` if tracking remains disabled.
    """
    if not tracking_cfg.get("enabled", False):
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_SILENT", "true")
        return None

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError:
        print(
            "[WARN] W&B tracking requested but the 'wandb' package is missing. "
            "Install wandb or disable tracking in config.tracking.enabled."
        )
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None

    project = tracking_cfg.get("project", "pcb-defect-detection")
    entity = tracking_cfg.get("entity")
    base_tags = list(tracking_cfg.get("tags", []))
    if tags:
        base_tags.extend(tags)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config or {},
        tags=base_tags or None,
        reinit=True,
    )

    return run


def finish_wandb_run(run) -> None:
    """Finish the provided W&B run if available."""
    if run is None:
        return

    try:
        run.finish()
    except Exception as exc:  # pragma: no cover - safety, shouldn't trigger normally
        print(f"[WARN] Failed to close W&B run cleanly: {exc}")
