from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_dynamic_capture_report_and_latest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 185)
    _write_image(images_train / "curve_lux80_0001-1.png", 110)
    _write_image(images_train / "curve_lux40_0001-1.png", 55)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_dynamic_capture"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/pixel/dynamic-capture",
        json={"workspace_root": str(workspace_root), "target_mid_lux": 100},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_groups"] >= 1
    assert payload["recommended_global_mode"] in {"forensic_wdr_bracket_3", "adaptive_bracket_capture", "single_capture"}
    assert Path(payload["report_json_path"]).exists()

    latest = client.get(
        "/api/research/v1/pixel/dynamic-capture/latest",
        params={"workspace_root": str(workspace_root)},
    )
    assert latest.status_code == 200
    assert latest.json()["target_mid_lux"] == 100
