import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pcb_defect_detection.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "PCB Defect Detection CLI" in result.stdout
    assert "benchmark-onnx" in result.stdout