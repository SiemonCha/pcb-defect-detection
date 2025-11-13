"""Namespace package wiring for the flattened source layout.

The project keeps feature modules directly under ``src/`` (for example
``src/analysis``).  To preserve the ``pcb_defect_detection.*`` import path used by
scripts, documentation, and tests, we alias those top-level packages into this
namespace at import time.  This keeps backwards compatibility without
re-nesting the source tree.
"""

from importlib import import_module
import sys
from types import ModuleType


def _alias(name: str, target: str) -> ModuleType:
    """Expose ``target`` as ``pcb_defect_detection.<name>``."""
    module = import_module(target)
    sys.modules[f"{__name__}.{name}"] = module
    return module


analysis = _alias("analysis", "analysis")
benchmarks = _alias("benchmarks", "benchmarks")
data = _alias("data", "data")
deployment = _alias("deployment", "deployment")
evaluation = _alias("evaluation", "evaluation")
setup = _alias("setup", "setup")
training = _alias("training", "training")
tracking = _alias("tracking", "tracking")
config = import_module(f"{__name__}.config")
cli = import_module(f"{__name__}.cli")

__all__ = [
    "analysis",
    "benchmarks",
    "cli",
    "config",
    "data",
    "deployment",
    "evaluation",
    "setup",
    "tracking",
    "training",
]
