from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_forensic_wdr_pipeline_surface(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 185)
    _write_image(images_train / "curve_lux80_0001-1.png", 110)
    _write_image(images_train / "curve_lux40_0001-1.png", 55)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_forensic_wdr"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/workspace/run-full",
        json={"workspace_root": str(workspace_root), "include_training_plan": True, "training_dry_run": True},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/pixel/forensic-wdr/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "overwrite": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed_groups"] >= 1
    assert Path(payload["outputs"][0]["output_path"]).exists()

    pixel_lab = client.post(
        "/api/research/v1/pixel/lab",
        params={"workspace_root": str(workspace_root), "target_lux": 100},
    )
    assert pixel_lab.status_code == 200
    assert any(method["method_name"] == "Forensic WDR" for method in pixel_lab.json()["methods"])

    live_monitor = client.post(
        "/api/research/v1/monitor/live",
        params={"workspace_root": str(workspace_root)},
    )
    assert live_monitor.status_code == 200
    assert live_monitor.json()["forensic_wdr_output_count"] >= 1
