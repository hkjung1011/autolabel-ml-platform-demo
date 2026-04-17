from __future__ import annotations

from json import JSONDecodeError
import json
import os
from pathlib import Path
import sys

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.core.config import runtime_is_frozen_bundle, settings
from app.domain.models import UploadAssetResponse
from app.domain.research_models import (
    AblationRunRequest,
    AutoLabelBuildRequest,
    BootstrapV1Request,
    CommercializationPlanRequest,
    CommercialStageRequest,
    DatasetDiscoveryRequest,
    DynamicCaptureRequest,
    ForensicWdrRunRequest,
    TargetLuxRunRequest,
    SegmentationBootstrapRequest,
    ReviewQueueUpdateRequest,
    EvidenceRunRequest,
    EvaluationRunRequest,
    FusionRunRequest,
    RegistrationVerifyRequest,
    RetinexRunRequest,
    SourceCatalogRequest,
    StageCandidateRequest,
    TestEvaluationRequest,
    TrainingRunRequest,
    WorkspacePipelineRunRequest,
)
from app.domain.defect_autolabel_models import (
    DefectAutolabelProjectRequest,
    DefectAutolabelRunRequest,
    DefectExportRequest,
    DefectReviewUpdateRequest,
)
from app.services.autolabel import autolabel_service
from app.services.benchmark import benchmark_service
from app.services.accuracy_audit import accuracy_audit_service
from app.services.commercialization import commercialization_service
from app.services.data_quality_audit import data_quality_audit_service
from app.services.desktop_package import desktop_package_service
from app.services.desktop_runtime import desktop_runtime_service
from app.services.evaluation import evaluation_service
from app.services.operator_guide import operator_guide_service
from app.services.fusion_runner import fusion_runner_service
from app.services.mask_bootstrap import mask_bootstrap_service
from app.services.live_monitor import live_monitor_service
from app.services.dynamic_capture import dynamic_capture_service
from app.services.defect_autolabel_runner import defect_autolabel_runner_service
from app.services.defect_export import defect_export_service
from app.services.defect_review import defect_review_service
from app.services.forensic_wdr import forensic_wdr_service
from app.services.pixel_lab import pixel_lab_service
from app.services.pipeline import pipeline_service
from app.services.program_status import program_status_service
from app.services.reporting import reporting_service
from app.services.registration import registration_service
from app.services.research import research_workspace_service
from app.services.review_queue import review_queue_service
from app.services.retinex_runner import retinex_runner_service
from app.services.target_lux_runner import target_lux_runner_service
from app.services.training import training_service
from app.services.workspace_runner import workspace_runner_service

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "pid": os.getpid(),
        "version": settings.version,
        "frozen": runtime_is_frozen_bundle(),
        "executable": sys.executable,
        "build_label": os.environ.get("DEFECT_VISION_BUILD_LABEL", "api"),
        "port": os.environ.get("DEFECT_VISION_PORT"),
    }


@router.post("/desktop/shutdown")
def desktop_shutdown(request: Request, background_tasks: BackgroundTasks) -> dict[str, object]:
    shutdown_callback = getattr(request.app.state, "desktop_shutdown_callback", None)
    if not callable(shutdown_callback):
        raise HTTPException(status_code=409, detail="Desktop shutdown is unavailable in this runtime mode.")
    background_tasks.add_task(shutdown_callback)
    return {"status": "shutdown_requested"}


@router.get("/dashboard")
def dashboard():
    return pipeline_service.dashboard_summary()


@router.post("/assets/demo-seed")
def demo_seed():
    return pipeline_service.seed_demo_assets()


@router.get("/assets")
def list_assets():
    return pipeline_service.list_assets()


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    try:
        return pipeline_service.get_asset(asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc


@router.post("/assets/upload", response_model=UploadAssetResponse)
async def upload_asset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    stored_path = settings.upload_dir / f"upload_{Path(file.filename or 'asset').stem}{suffix}"
    content = await file.read()
    stored_path.write_bytes(content)
    asset = pipeline_service.create_asset_from_upload(stored_path, Path(file.filename or "uploaded_asset").stem)
    return UploadAssetResponse(asset=asset, message="Asset uploaded.")


@router.post("/pipeline/run/{asset_id}")
def run_pipeline(asset_id: str):
    try:
        return pipeline_service.run_pipeline(asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc


@router.post("/research/v1/bootstrap")
def bootstrap_research_v1(request: BootstrapV1Request):
    try:
        return research_workspace_service.bootstrap_v1(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/latest")
def latest_research_v1():
    latest = research_workspace_service.latest()
    if latest is None:
        raise HTTPException(status_code=404, detail="No V1 workspace has been created yet.")
    return latest


@router.post("/research/v1/discovery/candidates")
def discover_research_candidates(request: DatasetDiscoveryRequest):
    try:
        return research_workspace_service.discover_candidates(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/discovery/stage")
def stage_research_candidate(request: StageCandidateRequest):
    try:
        return research_workspace_service.stage_candidate(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/commercial/source-catalog")
def commercial_source_catalog_v1(request: SourceCatalogRequest):
    try:
        return commercialization_service.build_source_catalog(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/commercial/source-catalog/latest")
def commercial_source_catalog_latest_v1(workspace_root: str):
    try:
        return commercialization_service.load_source_catalog(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/commercial/plan")
def commercial_plan_v1(request: CommercializationPlanRequest):
    try:
        return commercialization_service.build_plan(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/commercial/plan/latest")
def commercial_plan_latest_v1(workspace_root: str):
    try:
        return commercialization_service.load_plan(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/commercial/stage-protected")
def commercial_stage_protected_v1(request: CommercialStageRequest):
    try:
        return commercialization_service.stage_protected_source(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/retinex/run")
def run_retinex_v1(request: RetinexRunRequest):
    try:
        return retinex_runner_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/retinex/status/{job_id}")
def retinex_status(job_id: str):
    try:
        return retinex_runner_service.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Retinex job not found") from exc


@router.post("/research/v1/pixel/target-lux/run")
def run_target_lux_v1(request: TargetLuxRunRequest):
    try:
        return target_lux_runner_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/pixel/forensic-wdr/run")
def run_forensic_wdr_v1(request: ForensicWdrRunRequest):
    try:
        return forensic_wdr_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/pixel/forensic-wdr/status/{job_id}")
def forensic_wdr_status(job_id: str):
    try:
        return forensic_wdr_service.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Forensic WDR job not found") from exc


@router.get("/research/v1/pixel/target-lux/status/{job_id}")
def target_lux_status(job_id: str):
    try:
        return target_lux_runner_service.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Target-lux job not found") from exc


@router.post("/research/v1/pixel/dynamic-capture")
def dynamic_capture_v1(request: DynamicCaptureRequest):
    try:
        return dynamic_capture_service.build_report(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/pixel/dynamic-capture/latest")
def dynamic_capture_latest_v1(workspace_root: str):
    try:
        return dynamic_capture_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/pixel/lab")
def pixel_lab_v1(workspace_root: str, target_lux: int = 100):
    try:
        return pixel_lab_service.build_report(workspace_root, target_lux=target_lux)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/pixel/lab/latest")
def pixel_lab_latest_v1(workspace_root: str):
    try:
        return pixel_lab_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/registration/verify")
def verify_registration_v1(request: RegistrationVerifyRequest):
    try:
        return registration_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/registration/status/{job_id}")
def registration_status(job_id: str):
    try:
        return registration_service.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Registration job not found") from exc


@router.get("/research/v1/registration/report")
def registration_report(workspace_root: str, variant_source: str = "retinex_msrcr"):
    try:
        return registration_service.load_report(workspace_root=workspace_root, variant_source=variant_source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/fusion/mertens/run")
def run_mertens_v1(request: FusionRunRequest):
    try:
        return fusion_runner_service.run_mertens(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/fusion/daf/run")
def run_daf_v1(request: FusionRunRequest):
    try:
        return fusion_runner_service.run_daf(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/fusion/status/{job_id}")
def fusion_status(job_id: str):
    try:
        return fusion_runner_service.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Fusion job not found") from exc


@router.post("/research/v1/evaluation/readiness")
def evaluation_readiness_v1(request: EvaluationRunRequest):
    try:
        return evaluation_service.build_readiness_report(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/evaluation/latest")
def evaluation_readiness_latest(workspace_root: str):
    try:
        return evaluation_service.load_readiness_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/evidence/benchmark")
def evidence_benchmark_v1(request: EvidenceRunRequest):
    try:
        return benchmark_service.build_evidence_report(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/evidence/latest")
def evidence_benchmark_latest(workspace_root: str):
    try:
        return benchmark_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/evaluation/train")
def training_run_v1(request: TrainingRunRequest):
    try:
        return training_service.run_training(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/evaluation/train/start")
def training_start_v1(request: TrainingRunRequest):
    try:
        return training_service.start_training_job(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/evaluation/train/status/{job_id}")
def training_status_v1(job_id: str):
    try:
        return training_service.get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc


@router.get("/research/v1/evaluation/train/runs")
def training_runs_v1(workspace_root: str):
    try:
        return training_service.list_training_runs(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/evaluation/train/test-eval")
def training_test_eval_v1(request: TestEvaluationRequest):
    try:
        return training_service.run_test_evaluation(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/evaluation/train/test-eval/latest")
def training_test_eval_latest_v1(run_dir: str):
    try:
        return training_service.load_test_evaluation(run_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/reporting/scorecard")
def reporting_scorecard_v1(workspace_root: str):
    try:
        return reporting_service.build_scorecard(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/scorecard/latest")
def reporting_scorecard_latest_v1(workspace_root: str):
    try:
        return reporting_service.load_scorecard(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/arm-comparison")
def reporting_arm_comparison_v1(workspace_root: str):
    try:
        return reporting_service.build_arm_comparison(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/arm-comparison/latest")
def reporting_arm_comparison_latest_v1(workspace_root: str):
    try:
        return reporting_service.load_arm_comparison(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/export-csv")
def reporting_export_csv_v1(workspace_root: str):
    try:
        return reporting_service.export_csv_bundle(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/paper-pack")
def reporting_paper_pack_v1(workspace_root: str):
    try:
        return reporting_service.build_paper_pack(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/paper-pack/latest")
def reporting_paper_pack_latest_v1(workspace_root: str):
    try:
        return reporting_service.load_paper_pack(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/accuracy-audit")
def reporting_accuracy_audit_v1(workspace_root: str):
    try:
        return accuracy_audit_service.build_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/accuracy-audit/latest")
def reporting_accuracy_audit_latest_v1(workspace_root: str):
    try:
        return accuracy_audit_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/data-quality-audit")
def reporting_data_quality_audit_v1(workspace_root: str):
    try:
        return data_quality_audit_service.build_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/data-quality-audit/latest")
def reporting_data_quality_audit_latest_v1(workspace_root: str):
    try:
        return data_quality_audit_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/autolabel/bootstrap")
def autolabel_bootstrap_v1(request: AutoLabelBuildRequest):
    try:
        return autolabel_service.build_bootstrap_dataset(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/autolabel/latest")
def autolabel_latest_v1(workspace_root: str):
    try:
        return autolabel_service.load_bootstrap_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/segmentation/bootstrap")
def segmentation_bootstrap_v1(request: SegmentationBootstrapRequest):
    try:
        return mask_bootstrap_service.build_dataset(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/segmentation/latest")
def segmentation_bootstrap_latest_v1(workspace_root: str):
    try:
        return mask_bootstrap_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/review-queue/build")
def review_queue_build_v1(workspace_root: str):
    try:
        return review_queue_service.build_queue(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/review-queue/latest")
def review_queue_latest_v1(workspace_root: str):
    try:
        return review_queue_service.load_queue(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/review-queue/update")
def review_queue_update_v1(request: ReviewQueueUpdateRequest):
    try:
        return review_queue_service.update_queue_item(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/review-queue/export-approved")
def review_queue_export_approved_v1(workspace_root: str):
    try:
        return review_queue_service.export_approved_dataset(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/desktop/package-plan")
def desktop_package_plan_v1(workspace_root: str):
    try:
        return desktop_package_service.build_plan(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/desktop/package-plan/latest")
def desktop_package_plan_latest_v1(workspace_root: str):
    try:
        return desktop_package_service.load_plan(workspace_root)
    except (FileNotFoundError, ValidationError, JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/desktop/runtime-check")
def desktop_runtime_check_v1(workspace_root: str):
    try:
        return desktop_runtime_service.build_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/desktop/runtime-check/latest")
def desktop_runtime_check_latest_v1(workspace_root: str):
    try:
        return desktop_runtime_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/operator/guide")
def operator_guide_v1(workspace_root: str):
    try:
        return operator_guide_service.build_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/operator/guide/latest")
def operator_guide_latest_v1(workspace_root: str):
    try:
        return operator_guide_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/monitor/live")
def live_monitor_v1(workspace_root: str):
    try:
        return live_monitor_service.build_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/monitor/live/latest")
def live_monitor_latest_v1(workspace_root: str):
    try:
        return live_monitor_service.load_report(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/reporting/program-status")
def reporting_program_status_v1(workspace_root: str):
    try:
        return program_status_service.build_report(workspace_root)
    except (FileNotFoundError, ValidationError, JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/research/v1/reporting/program-status/latest")
def reporting_program_status_latest_v1(workspace_root: str):
    try:
        return program_status_service.load_report(workspace_root)
    except (FileNotFoundError, ValidationError, JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v1/evaluation/ablation")
def ablation_run_v1(request: AblationRunRequest):
    try:
        return training_service.run_ablation(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v1/workspace/run-full")
def workspace_run_full_v1(request: WorkspacePipelineRunRequest):
    try:
        return workspace_runner_service.run_full_pipeline(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v1/pipeline/latest")
def pipeline_latest(workspace_root: str):
    report_path = Path(workspace_root) / "evaluations" / "pipeline" / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing pipeline report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


@router.get("/research/v1/artifacts/preview")
def preview_artifact(path: str):
    artifact_path = Path(path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    if artifact_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}:
        raise HTTPException(status_code=400, detail="Preview only supports image artifacts.")
    return FileResponse(artifact_path)


@router.get("/research/v1/artifacts/download")
def download_artifact(path: str):
    artifact_path = Path(path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    return FileResponse(artifact_path, filename=artifact_path.name)


@router.post("/research/v2/defect-autolabel/project/init")
def defect_autolabel_project_init_v2(request: DefectAutolabelProjectRequest):
    try:
        return defect_autolabel_runner_service.init_project(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v2/defect-autolabel/run")
def defect_autolabel_run_v2(request: DefectAutolabelRunRequest):
    try:
        return defect_autolabel_runner_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/research/v2/defect-autolabel/latest")
def defect_autolabel_latest_v2(workspace_root: str):
    try:
        return defect_autolabel_runner_service.load_latest(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v2/defect-autolabel/review/build")
def defect_autolabel_review_build_v2(workspace_root: str):
    try:
        return defect_review_service.build_queue(workspace_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/v2/defect-autolabel/review/update")
def defect_autolabel_review_update_v2(request: DefectReviewUpdateRequest):
    try:
        return defect_review_service.update_queue_item(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research/v2/defect-autolabel/export")
def defect_autolabel_export_v2(request: DefectExportRequest):
    try:
        return defect_export_service.export_dataset(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
