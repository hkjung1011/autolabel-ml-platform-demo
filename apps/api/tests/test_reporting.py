from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_scorecard_and_csv_export(tmp_path: Path) -> None:
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

    scorecard_response = client.post(
        "/api/research/v1/reporting/scorecard",
        params={"workspace_root": str(workspace_root)},
    )
    assert scorecard_response.status_code == 200
    scorecard_payload = scorecard_response.json()
    assert scorecard_payload["research_score"] >= 1
    assert Path(scorecard_payload["report_json_path"]).exists()

    program_status_response = client.post(
        "/api/research/v1/reporting/program-status",
        params={"workspace_root": str(workspace_root)},
    )
    assert program_status_response.status_code == 200
    program_status_payload = program_status_response.json()
    assert program_status_payload["overall_progress_percent"] >= 1
    assert program_status_payload["autolabel_progress_percent"] >= 1
    assert len(program_status_payload["structure_items"]) >= 1
    assert Path(program_status_payload["report_json_path"]).exists()

    latest_program_status_response = client.get(
        "/api/research/v1/reporting/program-status/latest",
        params={"workspace_root": str(workspace_root)},
    )
    assert latest_program_status_response.status_code == 200
    assert latest_program_status_response.json()["current_stage"]

    accuracy_response = client.post(
        "/api/research/v1/reporting/accuracy-audit",
        params={"workspace_root": str(workspace_root)},
    )
    assert accuracy_response.status_code == 200
    accuracy_payload = accuracy_response.json()
    assert accuracy_payload["detection_ready"] is True
    assert accuracy_payload["segmentation_ready"] is False
    assert Path(accuracy_payload["report_json_path"]).exists()

    data_quality_response = client.post(
        "/api/research/v1/reporting/data-quality-audit",
        params={"workspace_root": str(workspace_root)},
    )
    assert data_quality_response.status_code == 200
    data_quality_payload = data_quality_response.json()
    assert data_quality_payload["registered_groups_scanned"] >= 1
    assert Path(data_quality_payload["report_json_path"]).exists()

    comparison_response = client.post(
        "/api/research/v1/reporting/arm-comparison",
        params={"workspace_root": str(workspace_root)},
    )
    assert comparison_response.status_code == 200
    comparison_payload = comparison_response.json()
    assert comparison_payload["deploy_candidate"] is not None

    paper_response = client.post(
        "/api/research/v1/reporting/paper-pack",
        params={"workspace_root": str(workspace_root)},
    )
    assert paper_response.status_code == 200
    paper_payload = paper_response.json()
    assert paper_payload["paper_readiness_score"] >= 1
    assert Path(paper_payload["report_json_path"]).exists()
    assert Path(paper_payload["ablation_csv_path"]).exists()

    export_response = client.post(
        "/api/research/v1/reporting/export-csv",
        params={"workspace_root": str(workspace_root)},
    )
    assert export_response.status_code == 200
    export_payload = export_response.json()
    assert Path(export_payload["files"]["scorecard_csv"]).exists()
    assert Path(export_payload["files"]["training_runs_csv"]).exists()
    assert Path(export_payload["files"]["arm_comparison_csv"]).exists()
    assert Path(export_payload["files"]["paper_pack_summary_csv"]).exists()
    assert Path(export_payload["files"]["paper_pack_ablation_csv"]).exists()
    assert Path(export_payload["files"]["accuracy_audit_csv"]).exists()
    assert Path(export_payload["files"]["data_quality_audit_csv"]).exists()
