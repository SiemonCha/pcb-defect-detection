import os
from pathlib import Path

import pytest

from data import resolve_dataset_yaml


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("PCB_DATASET", raising=False)


@pytest.fixture
def config_stub(monkeypatch, tmp_path):
    cache_file = tmp_path / "dataset_path.txt"
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    fallback_yaml = fallback_dir / "data.yaml"
    fallback_yaml.write_text("names: []\n", encoding="utf-8")

    def _stub_load_config():
        return {
            "data": {
                "dataset_path_file": str(cache_file),
                "fallback_patterns": ["fallback/data.yaml"],
            }
        }

    monkeypatch.setattr("data.utils.load_config", _stub_load_config)
    return cache_file, fallback_yaml


def test_env_variable_has_priority(monkeypatch, tmp_path, config_stub):
    env_yaml = tmp_path / "env_dataset.yaml"
    env_yaml.write_text("names: []\n", encoding="utf-8")
    monkeypatch.setenv("PCB_DATASET", str(env_yaml))

    cache_file, _ = config_stub

    resolved = resolve_dataset_yaml(update_cache=True)
    assert resolved == env_yaml.resolve()
    assert cache_file.read_text(encoding="utf-8").strip() == str(env_yaml.resolve())


def test_cache_file_is_used(tmp_path, config_stub):
    cache_file, fallback_yaml = config_stub
    cached_path = tmp_path / "cached_dataset.yaml"
    cached_path.write_text("names: []\n", encoding="utf-8")
    cache_file.write_text(str(cached_path), encoding="utf-8")

    resolved = resolve_dataset_yaml(update_cache=False)
    assert resolved == cached_path.resolve()
    assert cache_file.read_text(encoding="utf-8").strip() == str(cached_path)


def test_fallback_patterns_are_scanned(monkeypatch, tmp_path, config_stub):
    monkeypatch.chdir(tmp_path)
    cache_file, fallback_yaml = config_stub

    cache_file.write_text("", encoding="utf-8")

    resolved = resolve_dataset_yaml(update_cache=True)
    assert resolved == fallback_yaml.resolve()
    assert cache_file.read_text(encoding="utf-8").strip() == str(fallback_yaml.resolve())


def test_missing_dataset_raises(monkeypatch, config_stub):
    cache_file, _ = config_stub
    cache_file.unlink(missing_ok=True)
    monkeypatch.setattr("data.utils._scan_patterns", lambda patterns: None)

    with pytest.raises(FileNotFoundError):
        resolve_dataset_yaml(update_cache=False)
