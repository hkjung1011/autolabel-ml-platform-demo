from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_segmentation_bootstrap_builds_coarse_masks(tmp_path: Path) -> None:
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
    assert client.post(
        "/api/research/v1/autolabel/bootstrap",
        json={"workspace_root": str(workspace_root), "include_arms": ["raw160", "retinex", "mertens", "daf"], "overwrite": True},
    ).status_code == 200

    bootstrap_response = client.post(
        "/api/research/v1/segmentation/bootstrap",
        json={
            "workspace_root": str(workspace_root),
            "source_dataset_name": "bootstrap_merged",
            "include_splits": ["train", "val", "test"],
            "overwrite": True,
            "padding_ratio": 0.02,
            "min_padding_px": 1,
        },
    )
    assert bootstrap_response.status_code == 200
    payload = bootstrap_response.json()
    assert payload["total_items"] >= 1
    assert payload["bootstrap_mode"] == "coarse_box_mask"
    assert Path(payload["dataset_yaml_path"]).exists()
    assert Path(payload["mask_root"]).exists()
    assert Path(payload["sample_mask_paths"][0]).exists()

    latest_response = client.get(
        "/api/research/v1/segmentation/latest",
        params={"workspace_root": str(workspace_root)},
    )
    assert latest_response.status_code == 200
    assert latest_response.json()["total_items"] == payload["total_items"]

    accuracy_response = client.post(
        "/api/research/v1/reporting/accuracy-audit",
        params={"workspace_root": str(workspace_root)},
    )
    assert accuracy_response.status_code == 200
    accuracy_payload = accuracy_response.json()
    assert accuracy_payload["segmentation_bootstrap_ready"] is True
    assert accuracy_payload["segmentation_ready"] is False
    assert accuracy_payload["segmentation_bootstrap_dataset_root"]
    assert accuracy_payload["segmentation_bootstrap_mode"] == "coarse_box_mask"
    assert accuracy_payload["segmentation_bootstrap_refined_items"] == 0

    status_response = client.post(
        "/api/research/v1/reporting/program-status",
        params={"workspace_root": str(workspace_root)},
    )
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["segmentation_progress_percent"] >= 35
