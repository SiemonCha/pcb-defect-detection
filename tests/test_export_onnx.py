from pathlib import Path

import builtins

from pcb_defect_detection.deployment import export_onnx


def test_ensure_onnx_dependencies_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"onnxruntime", "onnxslim"}:
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, providers = export_onnx.ensure_onnx_dependencies("CPU")
    assert not ok
    assert providers == []


def test_run_export_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    weights_dir = tmp_path / "runs/train/test_run/weights"
    weights_dir.mkdir(parents=True)
    model_path = weights_dir / "best.pt"
    model_path.write_bytes(b"torch")

    onnx_path = weights_dir / "best.onnx"

    class DummyYOLO:
        def __init__(self, path):
            assert Path(path).exists()

        def __call__(self, *args, **kwargs):
            return None

        def export(self, format="onnx", simplify=True):
            onnx_path.write_bytes(b"onnx")
            return str(onnx_path)

    dummy_stats = {"mean": 10.0, "std": 1.0, "min": 9.0, "max": 11.0}

    monkeypatch.setattr(export_onnx, "YOLO", DummyYOLO)
    monkeypatch.setattr(export_onnx, "benchmark_inference", lambda *args, **kwargs: dummy_stats)
    monkeypatch.setattr(export_onnx, "ensure_onnx_dependencies", lambda platform: (True, []))

    result = export_onnx.run_export(str(model_path), runs=5)
    assert result == 0
    assert (weights_dir / "benchmark_results.txt").exists()
