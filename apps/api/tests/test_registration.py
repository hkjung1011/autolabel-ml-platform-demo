from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_pattern_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 34, 38), outline=(20, 20, 20), fill=(brightness // 2, brightness // 2, brightness // 2))
    draw.line((8, 50, 56, 10), fill=(min(255, brightness + 30),) * 3, width=3)
    image.save(path)


def test_registration_verify_accepts_retinex_variant(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_pattern_image(images_train / "curve_lux160_0001-1.png", 150)
    _write_pattern_image(images_train / "curve_lux80_0001-1.png", 70)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.35 0.44 0.34 0.31\n", encoding="utf-8")

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

    retinex = client.post(
        "/api/research/v1/retinex/run",
        json={
            "workspace_root": str(workspace_root),
            "source_luxes": ["80"],
            "method": "msrcr",
            "overwrite": True,
        },
    )
    assert retinex.status_code == 200

    response = client.post(
        "/api/research/v1/registration/verify",
        json={
            "workspace_root": str(workspace_root),
            "variant_source": "retinex_msrcr",
            "source_luxes": ["80"],
            "max_shift_px": 6,
            "max_label_iou_drift": 0.1,
            "min_similarity": 0.1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reports"]
    assert payload["accepted_count"] >= 1
    report_path = Path(payload["report_path"])
    accepted_manifest_path = Path(payload["accepted_manifest_path"])
    assert report_path.exists()
    assert accepted_manifest_path.exists()
    assert payload["registered_dataset_root"] is not None
    registered_root = Path(payload["registered_dataset_root"])
    registered_images = list((registered_root / "images").rglob("*.png"))
    registered_labels = list((registered_root / "labels").rglob("*.txt"))
    assert registered_images
    assert registered_labels
