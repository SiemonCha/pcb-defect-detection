from pathlib import Path

import numpy as np

from pcb_defect_detection.evaluation.confusion import plot_confusion_matrix


def test_plot_confusion_matrix_without_background(tmp_path):
    matrix = np.array([[5, 1], [2, 7]], dtype=float)
    output = tmp_path / "cm_no_background.png"
    plot_confusion_matrix(matrix, ["a", "b"], output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_confusion_matrix_with_background(tmp_path):
    matrix = np.array(
        [
            [8, 1, 0],
            [2, 9, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    output = tmp_path / "cm_with_background.png"
    plot_confusion_matrix(matrix, ["x", "y"], output)
    assert output.exists()
    assert output.stat().st_size > 0
