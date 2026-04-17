from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_defect_like_image(path: Path, brightness: int, defect_tone: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (56, 56), (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 16, 36, 36), outline=(defect_tone, defect_tone, defect_tone), width=2)
    draw.line((12, 44, 42, 22), fill=(255 - defect_tone, 255 - defect_tone, 255 - defect_tone), width=3)
    draw.ellipse((26, 12, 46, 30), outline=(defect_tone // 2, defect_tone // 2, defect_tone // 2), width=2)
    image.save(path)


def test_daf_generates_output_and_debug_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_defect_like_image(images_train / "curve_lux160_0001-1.png", 170, 25)
    _write_defect_like_image(images_train / "curve_lux80_0001-1.png", 95, 60)
    _write_defect_like_image(images_train / "curve_lux40_0001-1.png", 48, 120)
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
        "/api/research/v1/fusion/daf/run",
        json={
            "workspace_root": str(workspace_root),
            "source_luxes": ["40", "80", "160"],
            "method": "daf",
            "overwrite": True,
            "emit_debug_artifacts": True,
            "params": {"alpha": 0.6, "high_frequency_sigma": 3.0},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["outputs"]
    output = payload["outputs"][0]
    assert Path(output["output_path"]).exists()
    assert output["artifact_paths"]
    assert Path(payload["dataset_root"]).exists()
