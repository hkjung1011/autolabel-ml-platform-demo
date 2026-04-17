from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_target_lux_and_pixel_lab(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 180)
    _write_image(images_train / "curve_lux80_0001-1.png", 110)
    _write_image(images_train / "curve_lux40_0001-1.png", 60)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_pixel"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/workspace/run-full",
        json={"workspace_root": str(workspace_root), "include_training_plan": True, "training_dry_run": True},
    ).status_code == 200

    target_response = client.post(
        "/api/research/v1/pixel/target-lux/run",
        json={
            "workspace_root": str(workspace_root),
            "target_lux": 100,
            "source_luxes": ["40", "80", "160"],
            "apply_clahe": True,
        },
    )
    assert target_response.status_code == 200
    target_payload = target_response.json()
    assert target_payload["processed_groups"] >= 1
    assert Path(target_payload["outputs"][0]["output_path"]).exists()

    pixel_lab_response = client.post(
        "/api/research/v1/pixel/lab",
        params={"workspace_root": str(workspace_root), "target_lux": 100},
    )
    assert pixel_lab_response.status_code == 200
    pixel_payload = pixel_lab_response.json()
    assert pixel_payload["target_lux_ready"] is True
    assert any(method["method_name"] == "Target-Lux 100" for method in pixel_payload["methods"])
    assert Path(pixel_payload["report_json_path"]).exists()

    latest_response = client.get(
        "/api/research/v1/pixel/lab/latest",
        params={"workspace_root": str(workspace_root)},
    )
    assert latest_response.status_code == 200
    assert latest_response.json()["target_lux_value"] == 100
