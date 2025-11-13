import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(custom_path: str | None = None) -> Dict[str, Any]:
    """Load the default configuration and apply optional overrides."""
    with resources.files(__package__).joinpath("default.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    path = custom_path or os.getenv("PCB_CONFIG")
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as stream:
            override = yaml.safe_load(stream) or {}
        _merge(config, override)

    return config


__all__ = ["load_config"]
