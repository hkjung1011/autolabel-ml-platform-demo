import json
from pathlib import Path
import sys
import time
import types

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.domain.research_models import TrainingRunRequest
from app.services.training import training_service


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (brightness, brightness, brightness)).save(path)


def test_training_plan_and_ablation(tmp_path: Path) -> None:
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
        "/api/research/v1/retinex/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["80"], "method": "msrcr", "overwrite": True},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/registration/verify",
        json={"workspace_root": str(workspace_root), "source_luxes": ["80"], "min_similarity": 0.0},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/fusion/mertens/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "mertens"},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/fusion/daf/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "method": "daf"},
    ).status_code == 200
    assert client.post(
        "/api/research/v1/pixel/forensic-wdr/run",
        json={"workspace_root": str(workspace_root), "source_luxes": ["40", "80", "160"], "overwrite": True},
    ).status_code == 200

    train_response = client.post(
        "/api/research/v1/evaluation/train",
        json={"workspace_root": str(workspace_root), "arm": "raw160", "dry_run": True},
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload["status"] == "planned"
    assert train_payload["requested_device"] == "0"
    assert train_payload["metrics"]["requested_device"] == "0"
    assert Path(train_payload["artifact_paths"]["plan_json_path"]).exists()
    assert Path(train_payload["artifact_paths"]["summary_markdown_path"]).exists()

    mixed_response = client.post(
        "/api/research/v1/evaluation/train",
        json={"workspace_root": str(workspace_root), "arm": "mixed_raw_forensic_wdr", "dry_run": True},
    )
    assert mixed_response.status_code == 200
    mixed_payload = mixed_response.json()
    assert mixed_payload["status"] == "planned"
    assert mixed_payload["dataset_yaml_path"].endswith("mixed_raw_forensic_wdr.yaml")
    assert any("forensic_wdr variants" in note for note in mixed_payload["notes"])

    readiness_before = client.post(
        "/api/research/v1/evaluation/readiness",
        json={"workspace_root": str(workspace_root), "include_arms": ["raw160", "retinex", "mertens", "daf", "mixed_raw_forensic_wdr"]},
    )
    assert readiness_before.status_code == 200
    assert readiness_before.json()["completion_percent"] >= 80

    ablation_response = client.post(
        "/api/research/v1/evaluation/ablation",
        json={"workspace_root": str(workspace_root), "arms": ["raw160", "mixed_raw_forensic_wdr"], "dry_run": True},
    )
    assert ablation_response.status_code == 200
    ablation_payload = ablation_response.json()
    assert len(ablation_payload["runs"]) == 2
    assert Path(ablation_payload["report_json_path"]).exists()
    assert Path(ablation_payload["report_markdown_path"]).exists()

    readiness_after = client.get("/api/research/v1/evaluation/latest", params={"workspace_root": str(workspace_root)})
    assert readiness_after.status_code == 200
    assert readiness_after.json()["completion_percent"] >= 80


def test_live_training_job_with_external_command(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 90)
    _write_image(images_train / "curve_lux40_0001-1.png", 40)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_live"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200

    command = f'"{sys.executable}" -c "import time; print(\'trainer-start\'); time.sleep(0.2); print(\'trainer-done\')"'
    start_response = client.post(
        "/api/research/v1/evaluation/train/start",
        json={
            "workspace_root": str(workspace_root),
            "arm": "raw160",
            "dry_run": False,
            "run_name": "raw160_live_demo",
            "trainer_command": command,
        },
    )
    assert start_response.status_code == 200
    payload = start_response.json()
    assert payload["status"] in {"running", "completed"}
    assert payload["requested_device"] == "0"
    assert payload["runtime_mode"] == "external_command"
    job_id = payload["job_id"]

    final_payload = payload
    for _ in range(10):
        time.sleep(0.2)
        status_response = client.get(f"/api/research/v1/evaluation/train/status/{job_id}")
        assert status_response.status_code == 200
        final_payload = status_response.json()
        if final_payload["status"] in {"completed", "failed"}:
            break

    assert final_payload["status"] == "completed"
    assert any("trainer-done" in line for line in final_payload["tail_lines"])

    results_csv = Path(final_payload["output_dir"]) / "results.csv"
    results_csv.write_text(
        "epoch,metrics/mAP50(B),metrics/recall(B)\n1,0.81,0.92\n",
        encoding="utf-8",
    )

    history_response = client.get(
        "/api/research/v1/evaluation/train/runs",
        params={"workspace_root": str(workspace_root)},
    )
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["total_runs"] >= 1
    target_run = next(run for run in history_payload["runs"] if run["run_name"] == "raw160_live_demo")
    assert target_run["metrics"]["metrics/mAP50(B)"] == 0.81
    assert target_run["metrics"]["metrics/recall(B)"] == 0.92


def test_segmentation_bootstrap_training_plan(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    _write_image(images_train / "curve_lux80_0001-1.png", 90)
    _write_image(images_train / "curve_lux40_0001-1.png", 40)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_seg"
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
    assert client.post(
        "/api/research/v1/segmentation/bootstrap",
        json={"workspace_root": str(workspace_root), "source_dataset_name": "bootstrap_merged", "overwrite": True},
    ).status_code == 200

    train_response = client.post(
        "/api/research/v1/evaluation/train",
        json={"workspace_root": str(workspace_root), "arm": "seg_bootstrap", "dry_run": True},
    )
    assert train_response.status_code == 200
    train_payload = train_response.json()
    assert train_payload["status"] == "planned"
    assert train_payload["dataset_yaml_path"].endswith("segmentation_bootstrap.yaml")
    assert "yolo segment train" in train_payload["command_preview"]
    assert train_payload["metrics"]["task"] == "segment"


def test_frozen_live_training_uses_embedded_runner_request(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset_paint_defect"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "curve_lux160_0001-1.png", 160)
    (labels_train / "curve_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace_frozen"
    assert client.post(
        "/api/research/v1/bootstrap",
        json={"dataset_path": str(dataset_root), "workspace_root": str(workspace_root), "materialize_workspace": True},
    ).status_code == 200

    request = TrainingRunRequest(
        workspace_root=str(workspace_root),
        arm="raw160",
        dry_run=False,
        run_name="frozen_live_test",
    )
    response, ultralytics_installed = training_service._prepare_training_response(request)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\fake\DefectVisionResearch.exe")

    command, runtime_mode = training_service._resolve_live_command(request, response, ultralytics_installed)

    assert runtime_mode == "frozen_embedded"
    assert isinstance(command, list)
    assert command[0] == r"C:\fake\DefectVisionResearch.exe"
    assert command[1] == "--embedded-train"
    embedded_request = Path(command[2])
    assert embedded_request.exists()
    payload = json.loads(embedded_request.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    assert payload["trainer_command"] is None
    assert payload["run_name"] == "frozen_live_test"


def test_test_eval_route_and_history_metrics(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace_eval"
    run_dir = workspace_root / "evaluations" / "training" / "raw160_eval_demo"
    weights_dir = run_dir / "run" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"fake")

    data_yaml = workspace_root / "datasets" / "yolo_baseline" / "meta" / "raw160.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {workspace_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 2",
                "names:",
                "  0: defect",
                "  1: highlight",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (run_dir / "run" / "args.yaml").write_text(f"data: {data_yaml.as_posix()}\n", encoding="utf-8")
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "run_name": "raw160_eval_demo",
                "arm": "raw160",
                "status": "completed",
                "dry_run": False,
                "output_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )

    class _FakeBox:
        map50 = 0.91
        map = 0.77
        mp = 0.88
        mr = 0.84
        ap50 = [0.95, 0.87]
        ap = [0.8, 0.74]

    class _FakeResults:
        box = _FakeBox()

    class _FakeYOLO:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def val(self, **kwargs):
            output_dir = Path(kwargs["project"]) / kwargs["name"]
            output_dir.mkdir(parents=True, exist_ok=True)
            return _FakeResults()

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    eval_response = client.post(
        "/api/research/v1/evaluation/train/test-eval",
        json={"run_dir": str(run_dir), "device": "0"},
    )
    assert eval_response.status_code == 200
    eval_payload = eval_response.json()
    assert eval_payload["status"] == "completed"
    assert eval_payload["metrics"]["mAP50"] == 0.91
    assert Path(eval_payload["results_json_path"]).exists()
    assert Path(eval_payload["summary_text_path"]).exists()
    assert len(eval_payload["per_class"]) == 2

    latest_response = client.get(
        "/api/research/v1/evaluation/train/test-eval/latest",
        params={"run_dir": str(run_dir)},
    )
    assert latest_response.status_code == 200
    assert latest_response.json()["metrics"]["precision"] == 0.88

    history_response = client.get(
        "/api/research/v1/evaluation/train/runs",
        params={"workspace_root": str(workspace_root)},
    )
    assert history_response.status_code == 200
    history_payload = history_response.json()
    target_run = next(run for run in history_payload["runs"] if run["run_name"] == "raw160_eval_demo")
    assert target_run["metrics"]["test/mAP50"] == 0.91
    assert target_run["metrics"]["test/precision"] == 0.88
    assert Path(target_run["files"]["test_eval_results_json_path"]).exists()
