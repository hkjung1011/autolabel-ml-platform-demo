from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int, accent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (60, 60), (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 18, 38, 40), outline=(accent, accent, accent), width=2)
    draw.line((18, 46, 44, 20), fill=(255 - accent, 255 - accent, 255 - accent), width=3)
    image.save(path)


def test_evidence_benchmark_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 170, 30)
    _write_image(images_train / "curve_lux80_0001-1.png", 95, 70)
    _write_image(images_train / "curve_lux40_0001-1.png", 48, 120)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.45 0.48 0.32 0.32\n", encoding="utf-8")

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
        "/api/research/v1/fusion/mertens/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "mertens"},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/fusion/daf/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "daf"},
    ).status_code == 200

    response = client.post(
        "/api/research/v1/evidence/benchmark",
        json={
            "workspace_root": str(workspace_root),
            "source_lux": "80",
            "compare_arms": ["raw80", "retinex80", "mertens", "daf"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["common_group_count"] >= 1
    assert payload["arms"]
    assert payload["recommended_arm"]
    assert payload["peak_arm"]
    assert payload["key_takeaways"]
    assert Path(payload["report_json_path"]).exists()
    assert Path(payload["report_markdown_path"]).exists()

    latest = client.get("/api/research/v1/evidence/latest", params={"workspace_root": str(workspace_root)})
    assert latest.status_code == 200
    assert latest.json()["recommended_arm"] == payload["recommended_arm"]
