from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def _write_anomaly_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 20, 63), fill=(18, 18, 18))
    draw.rectangle((44, 0, 63, 63), fill=(245, 245, 245))
    draw.rectangle((22, 22, 42, 42), fill=(140, 140, 140))
    image.save(path)


def test_autolabel_bootstrap_updates_program_status(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_anomaly_image(images_train / "curve_lux160_0001-1.png")
    _write_anomaly_image(images_train / "curve_lux80_0001-1.png")
    _write_anomaly_image(images_train / "curve_lux40_0001-1.png")
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

    autolabel_response = client.post(
        "/api/research/v1/autolabel/bootstrap",
        json={
            "workspace_root": str(workspace_root),
            "include_arms": ["raw160", "retinex", "mertens", "daf"],
            "overwrite": True,
            "focus_mode": "defect_and_lighting_anomaly",
            "include_lighting_anomalies": True,
            "dark_threshold": 40,
            "bright_threshold": 230,
            "min_region_area_ratio": 0.01,
        },
    )
    assert autolabel_response.status_code == 200
    payload = autolabel_response.json()
    assert payload["total_proposals"] >= 1
    assert payload["anomaly_box_count"] >= 1
    assert Path(payload["dataset_yaml_path"]).exists()
    assert Path(payload["report_json_path"]).exists()

    review_response = client.post(
        "/api/research/v1/review-queue/build",
        params={"workspace_root": str(workspace_root)},
    )
    assert review_response.status_code == 200
    review_payload = review_response.json()
    assert review_payload["total_items"] >= 1
    assert Path(review_payload["report_json_path"]).exists()
    proposal_id = review_payload["items"][0]["proposal_id"]

    update_response = client.post(
        "/api/research/v1/review-queue/update",
        json={
            "workspace_root": str(workspace_root),
            "proposal_id": proposal_id,
            "action": "approve",
            "review_owner": "qa_lead",
            "note": "approved for detector retraining",
        },
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["status_counts"]["approved"] >= 1
    assert update_payload["reviewed_count"] >= 1
    assert update_payload["items"][0]["review_owner"] == "qa_lead"

    export_response = client.post(
        "/api/research/v1/review-queue/export-approved",
        params={"workspace_root": str(workspace_root)},
    )
    assert export_response.status_code == 200
    export_payload = export_response.json()
    assert export_payload["exported_items"] >= 1
    assert Path(export_payload["approved_dataset_root"]).exists()
    assert Path(export_payload["report_json_path"]).exists()

    status_response = client.post(
        "/api/research/v1/reporting/program-status",
        params={"workspace_root": str(workspace_root)},
    )
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["autolabel_progress_percent"] >= 55
    assert any(item["module_name"] == "Auto-Label Loop" for item in status_payload["structure_items"])
    assert any(item["module_name"] == "Review Queue" for item in status_payload["structure_items"])
