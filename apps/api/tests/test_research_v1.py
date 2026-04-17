from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_bootstrap_research_v1(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"

    _write_image(images_train / "curve_lux160_0001-1.png", (120, 120, 120))
    _write_image(images_train / "curve_lux80_0001-1.png", (80, 80, 80))
    _write_image(images_train / "curve_lux40_0001-1.png", (40, 40, 40))
    _write_image(images_train / "curve_lux160_0002-1.png", (140, 140, 140))
    labels_train.mkdir(parents=True, exist_ok=True)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_train / "curve_lux160_0002-1.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    response = client.post(
        "/api/research/v1/bootstrap",
        json={
            "dataset_path": str(dataset_root),
            "workspace_root": str(workspace_root),
            "weights_path": str(tmp_path / "weights.pt"),
            "materialize_workspace": True,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["summary"]["total_images"] == 4
    assert payload["summary"]["labeled_anchor_count"] == 2
    assert payload["summary"]["labeled_with_both"] == 1
    assert payload["summary"]["frozen_split_counts"]
    assert payload["summary"]["labeled_split_counts"]

    pair_manifest = Path(payload["summary"]["artifact_paths"]["pair_manifest_path"])
    labeled_manifest = Path(payload["summary"]["artifact_paths"]["labeled_manifest_path"])
    split_manifest = Path(payload["summary"]["artifact_paths"]["split_manifest_path"])
    assert pair_manifest.exists()
    assert labeled_manifest.exists()
    assert split_manifest.exists()

    group_lookup = {group["key"]: group for group in payload["pair_groups"]}
    frozen_split = group_lookup["train|curve|0001-1"]["frozen_split"]
    baseline_image = workspace_root / "datasets" / "yolo_baseline" / "images" / frozen_split / "curve_lux160_0001-1.png"
    reuse_image = workspace_root / "datasets" / "yolo_cross_lux_reuse" / "images" / frozen_split / "curve_lux80_0001-1.png"
    reuse_label = workspace_root / "datasets" / "yolo_cross_lux_reuse" / "labels" / frozen_split / "curve_lux80_0001-1.txt"
    assert baseline_image.exists()
    assert reuse_image.exists()
    assert reuse_label.exists()
