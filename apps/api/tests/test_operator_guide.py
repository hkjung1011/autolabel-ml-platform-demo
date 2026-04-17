from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_operator_guide_builds_and_loads(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 90)
    _write_image(images_train / "curve_lux40_0001-1.png", 40)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/workspace/run-full",
        json={"workspace_root": str(workspace_root), "include_training_plan": True, "training_dry_run": True},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/operator/guide",
        params={"workspace_root": str(workspace_root)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["operator_readiness_score"] >= 1
    assert payload["steps"]
    assert Path(payload["report_json_path"]).exists()

    latest_response = client.get(
        "/api/research/v1/operator/guide/latest",
        params={"workspace_root": str(workspace_root)},
    )
    assert latest_response.status_code == 200
    assert latest_response.json()["primary_action"]

