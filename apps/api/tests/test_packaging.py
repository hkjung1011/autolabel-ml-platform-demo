from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_desktop_package_plan_is_reported(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 100)
    _write_image(images_train / "curve_lux40_0001-1.png", 50)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/desktop/package-plan",
        params={"workspace_root": str(workspace_root)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert Path(payload["entry_script_path"]).exists()
    assert Path(payload["build_script_path"]).exists()
    assert Path(payload["spec_path"]).exists()
    assert Path(payload["report_json_path"]).exists()
