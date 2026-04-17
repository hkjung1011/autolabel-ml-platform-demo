from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 14, 30, 32), outline=(20, 20, 20), width=2)
    image.save(path)


def test_evaluation_readiness_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 90)
    _write_image(images_train / "curve_lux40_0001-1.png", 50)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.45 0.45 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/retinex/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["80"], "method": "msrcr", "overwrite": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/registration/verify",
        json={"workspace_root": str(workspace_root), "source_luxes": ["80"], "min_similarity": 0.0},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/fusion/mertens/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "mertens"},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/fusion/daf/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "daf"},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/evaluation/readiness",
        json={"workspace_root": str(workspace_root), "include_arms": ["raw160", "retinex", "mertens", "daf"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["completion_percent"] >= 80
    assert payload["execution_readiness_percent"] <= payload["completion_percent"]
    assert payload["arms"]
    assert Path(payload["report_json_path"]).exists()
    assert Path(payload["report_markdown_path"]).exists()

    latest = client.get("/api/research/v1/evaluation/latest", params={"workspace_root": str(workspace_root)})
    assert latest.status_code == 200
    assert latest.json()["completion_percent"] == payload["completion_percent"]
