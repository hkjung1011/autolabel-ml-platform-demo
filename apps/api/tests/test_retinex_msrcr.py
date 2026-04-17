from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)


def test_retinex_run_generates_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", (140, 135, 130))
    _write_image(images_train / "curve_lux80_0001-1.png", (70, 68, 66))
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    bootstrap = client.post(
        "/api/research/v1/bootstrap",
        json={
            "dataset_path": str(dataset_root),
            "workspace_root": str(workspace_root),
            "materialize_workspace": True,
        },
    )
    assert bootstrap.status_code == 200

    response = client.post(
        "/api/research/v1/retinex/run",
        json={
            "workspace_root": str(workspace_root),
            "source_luxes": ["80"],
            "method": "msrcr",
            "overwrite": True,
            "params": {"sigma_list": [15, 80, 250], "gain": 128},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"completed", "partial"}
    assert payload["outputs"]
    output_path = Path(payload["outputs"][0]["output_path"])
    assert output_path.exists()
    assert payload["outputs"][0]["ssim_vs_anchor"] is not None
