from pcb_defect_detection.config import load_config


def test_load_config_override(tmp_path, monkeypatch):
    override = tmp_path / "custom.yaml"
    override.write_text(
        """
training:
  baseline:
    epochs: 5
        """
    )

    monkeypatch.setenv("PCB_CONFIG", str(override))
    cfg = load_config()
    assert cfg["training"]["baseline"]["epochs"] == 5
