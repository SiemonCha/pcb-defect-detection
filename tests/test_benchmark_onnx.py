from pathlib import Path

import numpy as np

from benchmarks.onnx import benchmark_onnx_model


class FakeInput:
    name = "images"
    shape = [1, 3, 640, 640]


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, inputs):
        self.calls += 1
        array = inputs["images"]
        assert isinstance(array, np.ndarray)
        assert array.shape == (1, 3, 320, 320)
        return []


def test_benchmark_onnx_model(monkeypatch, tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"dummy")

    fake_session = FakeSession()

    monkeypatch.setattr(
        "benchmarks.onnx._load_session",
        lambda model_path, providers=None: fake_session,
    )

    stats = benchmark_onnx_model(fake_model, runs=5, warmup=1, imgsz=320)
    assert stats["runs"] == 5
    assert stats["mean_ms"] >= 0
    assert fake_session.calls == 6  # warmup + runs
