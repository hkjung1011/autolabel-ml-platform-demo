from pathlib import Path
import json

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter

from app.main import app
from app.services.defect_inference import defect_inference_service
from app.services.defect_quality import defect_quality_service
from app.domain.defect_autolabel_models import DefectAssetRecord


client = TestClient(app)


def _write_defect_image(
    path: Path,
    *,
    size: tuple[int, int] = (64, 64),
    background: int = 140,
    blur_radius: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (background, background, background))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 12, 30, 30), fill=(40, 40, 40))
    draw.line((5, size[1] - 10, size[0] - 6, size[1] - 20), fill=(15, 15, 15), width=3)
    draw.rectangle((size[0] - 14, 4, size[0] - 4, 16), fill=(240, 240, 240))
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    image.save(path)


def test_defect_quality_assigns_expected_lux_bucket(tmp_path: Path) -> None:
    image_path = tmp_path / "dark_sample.png"
    _write_defect_image(image_path, background=25)
    metrics = defect_quality_service.build_metrics(asset_id="asset_dark", image_path=image_path)
    assert metrics.lux_bucket in {"very_dark", "dark"}


def test_direct_detection_uses_domain_specific_classes(tmp_path: Path) -> None:
    image_path = tmp_path / "line_sample.png"
    _write_defect_image(image_path, background=145)
    gray_metrics = defect_quality_service.build_metrics(asset_id="asset_line", image_path=image_path)
    asset = DefectAssetRecord(
        asset_id="asset_line",
        image_path=str(image_path),
        group_id="group_line",
        shot_id="shot001",
        reported_lux=80,
        estimated_lux_bucket=gray_metrics.lux_bucket,
        split="train",
        width=64,
        height=64,
    )

    ship = defect_inference_service.detect_asset(asset=asset, quality=gray_metrics.model_dump(), domain="ship_defect")
    metal = defect_inference_service.detect_asset(asset=asset, quality=gray_metrics.model_dump(), domain="metal_plate_defect")

    assert all(item.class_name != "scratch" for item in ship)
    assert all(item.class_name != "coating_damage" for item in metal)
    assert any(item.class_name in {"crack", "corrosion", "weld_defect"} for item in ship)
    assert any(item.class_name in {"scratch", "surface_stain", "burr"} for item in metal)


def test_anchor_selection_prefers_best_quality_not_max_lux(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_defect_image(dataset_root / "train" / "panel_shot001_lux40.png", background=55)
    _write_defect_image(dataset_root / "train" / "panel_shot001_lux80.png", background=135)
    _write_defect_image(dataset_root / "train" / "panel_shot001_lux160.png", background=240, blur_radius=2.0)

    workspace_root = tmp_path / "workspace_anchor"
    init_response = client.post(
        "/api/research/v2/defect-autolabel/project/init",
        json={
            "input_root": str(dataset_root),
            "workspace_root": str(workspace_root),
            "domain": "ship_defect",
            "dataset_mode": "paired_lux",
        },
    )
    assert init_response.status_code == 200

    run_response = client.post(
        "/api/research/v2/defect-autolabel/run",
        json={
            "workspace_root": str(workspace_root),
            "domain": "ship_defect",
            "run_mode": "full",
            "overwrite": True,
        },
    )
    assert run_response.status_code == 200

    group_manifest_path = workspace_root / "manifests" / "group_manifest.json"
    groups = json.loads(group_manifest_path.read_text(encoding="utf-8"))
    anchor_asset_id = groups[0]["anchor_asset_id"]
    assets = json.loads((workspace_root / "manifests" / "asset_manifest.json").read_text(encoding="utf-8"))
    anchor_image_path = next(item["image_path"] for item in assets if item["asset_id"] == anchor_asset_id)
    assert anchor_image_path.endswith("panel_shot001_lux80.png")


def test_v2_flow_exports_only_original_images_and_handles_failed_propagation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_flow"
    _write_defect_image(dataset_root / "train" / "shipA_shot100_lux80.png", size=(64, 64), background=138)
    _write_defect_image(dataset_root / "train" / "shipA_shot100_lux160.png", size=(92, 60), background=235)
    _write_defect_image(dataset_root / "val" / "shipB_shot101_lux80.png", size=(64, 64), background=145)

    workspace_root = tmp_path / "workspace_flow"
    init_response = client.post(
        "/api/research/v2/defect-autolabel/project/init",
        json={
            "input_root": str(dataset_root),
            "workspace_root": str(workspace_root),
            "domain": "ship_defect",
            "dataset_mode": "paired_lux",
        },
    )
    assert init_response.status_code == 200

    run_response = client.post(
        "/api/research/v2/defect-autolabel/run",
        json={
            "workspace_root": str(workspace_root),
            "domain": "ship_defect",
            "run_mode": "full",
            "overwrite": True,
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["proposal_count"] >= 1

    propagation_report = json.loads((workspace_root / "reports" / "propagation_summary.json").read_text(encoding="utf-8"))
    assert any(not item["accepted"] for item in propagation_report)

    review_items = json.loads((workspace_root / "review_queue" / "items.json").read_text(encoding="utf-8"))
    pending = [item for item in review_items if item["review_status"] == "pending"]
    if pending:
        update_response = client.post(
            "/api/research/v2/defect-autolabel/review/update",
            json={
                "workspace_root": str(workspace_root),
                "proposal_id": pending[0]["proposal_id"],
                "action": "approve",
                "review_owner": "qa",
                "note": "approved for export",
            },
        )
        assert update_response.status_code == 200

    export_response = client.post(
        "/api/research/v2/defect-autolabel/export",
        json={
            "workspace_root": str(workspace_root),
            "domain": "ship_defect",
            "overwrite": True,
        },
    )
    assert export_response.status_code == 200
    export_payload = export_response.json()
    assert export_payload["exported_images"] == 3
    assert Path(export_payload["dataset_yaml_path"]).exists()

    exported_images = sorted((workspace_root / "exports" / "yolo_detection" / "images").rglob("*.*"))
    exported_names = {path.name for path in exported_images}
    assert "shipA_shot100_lux80.png" in exported_names
    assert "shipA_shot100_lux160.png" in exported_names
    assert all("normalized" not in name and "edge" not in name for name in exported_names)
