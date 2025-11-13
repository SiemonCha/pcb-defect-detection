import builtins
from types import SimpleNamespace

import pytest

from tracking.wandb_integration import configure_wandb, finish_wandb_run


def test_configure_wandb_disabled(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    run = configure_wandb({"enabled": False}, run_name="test")
    assert run is None
    assert "WANDB_MODE" in builtins.__import__("os").environ


def test_configure_wandb_missing_dependency(monkeypatch):
    monkeypatch.setitem(builtins.__dict__, "__import__", builtins.__import__)

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    run = configure_wandb({"enabled": True}, run_name="test")
    assert run is None


def test_finish_wandb_run(monkeypatch):
    finished = False

    class Run(SimpleNamespace):
        def finish(self):
            nonlocal finished
            finished = True

    finish_wandb_run(Run())
    assert finished
