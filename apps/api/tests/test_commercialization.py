from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_commercial_source_catalog_and_plan(tmp_path: Path) -> None:
    scan_root = tmp_path / "external_drive"
    dataset_root = scan_root / "ship_defect_dataset"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 96)
    _write_image(images_train / "curve_lux40_0001-1.png", 48)
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

    source_catalog_response = client.post(
        "/api/research/v1/commercial/source-catalog",
        json={"scan_root": str(scan_root), "workspace_root": str(workspace_root), "limit": 8, "min_images": 1},
    )
    assert source_catalog_response.status_code == 200
    source_catalog_payload = source_catalog_response.json()
    assert source_catalog_payload["total_entries"] >= 1
    assert source_catalog_payload["protected_source_count"] >= 1
    assert "read-only" in source_catalog_payload["source_policy_summary"].lower()
    assert Path(source_catalog_payload["report_json_path"]).exists()

    commercial_plan_response = client.post(
        "/api/research/v1/commercial/plan",
        json={"workspace_root": str(workspace_root), "scan_root": str(scan_root), "refresh_source_catalog": True, "limit": 8, "min_images": 1},
    )
    assert commercial_plan_response.status_code == 200
    commercial_plan_payload = commercial_plan_response.json()
    assert commercial_plan_payload["commercial_stage"]
    assert commercial_plan_payload["protected_source_count"] >= 1
    assert Path(commercial_plan_payload["report_json_path"]).exists()

    stage_response = client.post(
        "/api/research/v1/commercial/stage-protected",
        json={
            "workspace_root": str(workspace_root),
            "source_dataset_root": str(dataset_root),
            "staged_name": "ship_defect_copy",
            "max_groups": 1,
            "prefer_labeled": True,
            "bootstrap_after_stage": True,
            "run_pipeline_after_stage": True,
            "include_training_plan": True,
            "training_dry_run": True,
        },
    )
    assert stage_response.status_code == 200
    stage_payload = stage_response.json()
    assert stage_payload["source_policy"] == "read_only"
    assert stage_payload["ingest_mode"] == "copy_only_stage"
    assert stage_payload["selected_group_count"] >= 1
    assert Path(stage_payload["staged_dataset_root"]).exists()
    assert stage_payload["staged_workspace_root"]
    assert Path(stage_payload["staged_workspace_root"]).exists()
