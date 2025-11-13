import json
from pathlib import Path

from pcb_defect_detection.setup import check_dataset


def _create_coco_file(base_dir: Path) -> Path:
    image_path = base_dir / "image1.jpg"
    image_path.write_bytes(b"fake")

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": 100,
                "height": 100,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            }
        ],
        "categories": [{"id": 0, "name": "defect"}],
    }

    coco_path = base_dir / "annotations.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path


def test_report_and_optionally_fix(monkeypatch, tmp_path):
    coco_path = _create_coco_file(tmp_path)

    monkeypatch.setattr(check_dataset, "find_coco_jsons", lambda: [str(coco_path)])

    exit_code = check_dataset.report_and_optionally_fix(auto_fix=True)
    assert exit_code == 0

    label_path = tmp_path / "image1.txt"
    assert label_path.exists()
    content = label_path.read_text().strip()
    assert content != ""
