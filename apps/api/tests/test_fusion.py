from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_exposure_image(path: Path, brightness: int, accent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 12, 30, 30), outline=(accent, accent, accent), fill=(brightness // 2, brightness // 2, brightness // 2))
    draw.ellipse((24, 20, 42, 38), outline=(255 - accent, 255 - accent, 255 - accent), width=2)
    image.save(path)


def test_mertens_fusion_generates_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_exposure_image(images_train / "curve_lux160_0001-1.png", 160, 20)
    _write_exposure_image(images_train / "curve_lux80_0001-1.png", 90, 50)
    _write_exposure_image(images_train / "curve_lux40_0001-1.png", 45, 90)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.45 0.45 0.3 0.3\n", encoding="utf-8")

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
        "/api/research/v1/fusion/mertens/run",
        json={
            "workspace_root": str(workspace_root),
            "source_luxes": ["40", "80", "160"],
            "method": "mertens",
            "overwrite": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["outputs"]
    output_path = Path(payload["outputs"][0]["output_path"])
    assert output_path.exists()
    assert payload["outputs"][0]["dynamic_range_score"] > 0
    dataset_root = Path(payload["dataset_root"])
    assert list((dataset_root / "images").rglob("*.png"))
    assert list((dataset_root / "labels").rglob("*.txt"))


def test_fusion_marks_missing_exposures_as_skipped(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_exposure_image(images_train / "curve_lux160_0001-1.png", 160, 20)
    _write_exposure_image(images_train / "curve_lux80_0001-1.png", 90, 50)
    _write_exposure_image(images_train / "curve_lux160_0002-1.png", 150, 35)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.45 0.45 0.3 0.3\n", encoding="utf-8")
    (labels_train / "curve_lux160_0002-1.txt").write_text("0 0.50 0.50 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_skipped"
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
        "/api/research/v1/fusion/mertens/run",
        json={
            "workspace_root": str(workspace_root),
            "source_luxes": ["40", "80", "160"],
            "method": "mertens",
            "overwrite": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert len(payload["outputs"]) == 1
    assert len(payload["skipped"]) == 1
    assert payload["errors"] == []
    skipped = payload["skipped"][0]
    assert skipped["reason"] == "insufficient_exposures"
    assert skipped["available_luxes"] == ["160"]
