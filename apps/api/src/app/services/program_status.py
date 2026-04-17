from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    AccuracyAuditResponse,
    EvidenceRunRequest,
    EvaluationRunRequest,
    ProgramStatusResponse,
    ProgramStructureItem,
    TrainingRunHistoryResponse,
)
from app.services.accuracy_audit import accuracy_audit_service
from app.services.benchmark import benchmark_service
from app.services.desktop_package import desktop_package_service
from app.services.evaluation import evaluation_service
from app.services.reporting import reporting_service
from app.services.review_queue import review_queue_service
from app.services.training import training_service


class ProgramStatusService:
    def build_report(self, workspace_root: str) -> ProgramStatusResponse:
        workspace_path = Path(workspace_root)
        readiness = self._load_or_build_readiness(workspace_path)
        evidence = self._load_or_build_evidence(workspace_path)
        scorecard = self._load_or_build_scorecard(workspace_path)
        accuracy = self._load_or_build_accuracy(workspace_path)
        comparison = self._load_or_build_comparison(workspace_path)
        paper_pack = self._load_or_build_paper_pack(workspace_path)
        package_plan = self._load_or_build_package_plan(workspace_path)
        training_runs = self._safe_training_runs(workspace_path)
        commercial_snapshot = self._load_commercial_snapshot(workspace_path)

        pipeline_report_path = workspace_path / "evaluations" / "pipeline" / "report.json"
        export_root = workspace_path / "evaluations" / "exports"
        autolabel_root = workspace_path / "datasets" / "autolabel"
        pipeline_ready = pipeline_report_path.exists()
        export_ready = export_root.exists() and any(export_root.glob("*.csv"))
        autolabel_ready = autolabel_root.exists()
        review_queue_ready = self._has_review_queue(workspace_path)

        detection_progress = self._detection_progress(accuracy, evidence.recommended_arm, comparison.deploy_candidate)
        autolabel_progress, autolabel_blockers = self._autolabel_progress(
            accuracy=accuracy,
            readiness_phase_status=readiness.phase_status,
            evidence_recommended_arm=evidence.recommended_arm,
            latest_metric_summary=scorecard.latest_metric_summary,
            autolabel_ready=autolabel_ready,
            review_queue_ready=review_queue_ready,
        )
        segmentation_progress = self._segmentation_progress(
            accuracy=accuracy,
            training_runs=training_runs,
            autolabel_ready=autolabel_ready,
        )
        reporting_progress = self._reporting_progress(
            scorecard_ready=Path(scorecard.report_json_path).exists(),
            accuracy_ready=Path(accuracy.report_json_path).exists(),
            comparison_ready=Path(comparison.report_json_path).exists(),
            paper_pack_ready=Path(paper_pack.report_json_path).exists(),
            pipeline_ready=pipeline_ready,
            export_ready=export_ready,
        )

        overall_progress = max(
            0,
            min(
                100,
                round(
                    (readiness.completion_percent * 0.35)
                    + (readiness.execution_readiness_percent * 0.20)
                    + (detection_progress * 0.15)
                    + (autolabel_progress * 0.15)
                    + (segmentation_progress * 0.05)
                    + (reporting_progress * 0.10)
                ),
            ),
        )

        structure_items = self._build_structure_items(
            workspace_path=workspace_path,
            readiness_phase_status=readiness.phase_status,
            accuracy=accuracy,
            scorecard=scorecard,
            pipeline_ready=pipeline_ready,
            export_ready=export_ready,
            paper_pack_ready=Path(paper_pack.report_json_path).exists(),
            packaging_ready=Path(package_plan.report_json_path).exists(),
            package_build_ready=package_plan.build_ready,
            detection_progress=detection_progress,
            autolabel_progress=autolabel_progress,
            segmentation_progress=segmentation_progress,
            autolabel_blockers=autolabel_blockers,
            review_queue_ready=review_queue_ready,
            commercial_snapshot=commercial_snapshot,
        )

        blockers = self._dedupe(
            list(scorecard.blockers)
            + list(accuracy.blockers)
            + autolabel_blockers
        )
        next_actions = self._next_actions(
            accuracy=accuracy,
            scorecard=scorecard,
            autolabel_progress=autolabel_progress,
            segmentation_progress=segmentation_progress,
            blockers=blockers,
        )
        current_stage = self._current_stage(
            detection_ready=accuracy.detection_ready,
            autolabel_progress=autolabel_progress,
            segmentation_ready=accuracy.segmentation_ready,
            field_score=scorecard.field_score,
        )
        summary_points = [
            f"Structure completion is {readiness.completion_percent}% and runnable readiness is {readiness.execution_readiness_percent}%.",
            f"Detection benchmarking is {detection_progress}% ready; segmentation is {segmentation_progress}% ready.",
            f"Auto-label loop progress is {autolabel_progress}%, with current recommended arm `{evidence.recommended_arm or 'n/a'}`.",
            f"Review queue is {'live' if review_queue_ready else 'not built yet'}, and desktop packaging is {'build-ready' if package_plan.build_ready else 'planned'}",
            f"Current stage: {current_stage}.",
        ]
        if commercial_snapshot is not None:
            summary_points.append(
                f"Commercial stage is {commercial_snapshot.get('commercial_stage', 'n/a')} with readiness {commercial_snapshot.get('commercial_readiness_score', 0)}/100 and {commercial_snapshot.get('protected_source_count', 0)} protected sources."
            )

        reporting_root = workspace_path / "evaluations" / "reporting"
        reporting_root.mkdir(parents=True, exist_ok=True)
        response = ProgramStatusResponse(
            workspace_root=str(workspace_path),
            overall_progress_percent=overall_progress,
            execution_readiness_percent=readiness.execution_readiness_percent,
            autolabel_progress_percent=autolabel_progress,
            detection_progress_percent=detection_progress,
            segmentation_progress_percent=segmentation_progress,
            current_stage=current_stage,
            recommended_arm=evidence.recommended_arm,
            deploy_candidate=comparison.deploy_candidate,
            commercial_stage=commercial_snapshot.get("commercial_stage") if commercial_snapshot else None,
            commercial_readiness_score=commercial_snapshot.get("commercial_readiness_score") if commercial_snapshot else None,
            protected_source_count=int(commercial_snapshot.get("protected_source_count", 0)) if commercial_snapshot else 0,
            staged_workspace_count=int(commercial_snapshot.get("staged_workspace_count", 0)) if commercial_snapshot else 0,
            research_score=scorecard.research_score,
            field_score=scorecard.field_score,
            production_score=scorecard.production_score,
            summary_points=summary_points,
            blockers=blockers,
            next_actions=next_actions,
            structure_items=structure_items,
            report_json_path=str(reporting_root / "program_status.json"),
            report_markdown_path=str(reporting_root / "program_status.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> ProgramStatusResponse:
        report_path = Path(workspace_root) / "evaluations" / "reporting" / "program_status.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing program status report: {report_path}")
        return ProgramStatusResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _load_or_build_readiness(self, workspace_path: Path):
        return evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=str(workspace_path),
                include_arms=list(evaluation_service.ARM_ROOTS.keys()),
                refresh_report=True,
            )
        )

    def _load_or_build_evidence(self, workspace_path: Path):
        try:
            return benchmark_service.load_report(workspace_path)
        except FileNotFoundError:
            return benchmark_service.build_evidence_report(
                EvidenceRunRequest(workspace_root=str(workspace_path), refresh_report=True)
            )

    def _load_or_build_scorecard(self, workspace_path: Path):
        return reporting_service.build_scorecard(str(workspace_path))

    def _load_or_build_accuracy(self, workspace_path: Path) -> AccuracyAuditResponse:
        return accuracy_audit_service.build_report(str(workspace_path))

    def _load_or_build_comparison(self, workspace_path: Path):
        return reporting_service.build_arm_comparison(str(workspace_path))

    def _load_or_build_paper_pack(self, workspace_path: Path):
        return reporting_service.build_paper_pack(str(workspace_path))

    def _load_or_build_package_plan(self, workspace_path: Path):
        return desktop_package_service.build_plan(str(workspace_path))

    def _safe_training_runs(self, workspace_path: Path) -> TrainingRunHistoryResponse:
        try:
            return training_service.list_training_runs(str(workspace_path))
        except FileNotFoundError:
            return TrainingRunHistoryResponse(workspace_root=str(workspace_path), total_runs=0, runs=[])

    def _has_review_queue(self, workspace_path: Path) -> bool:
        try:
            review_queue_service.load_queue(str(workspace_path))
            return True
        except FileNotFoundError:
            return False

    def _detection_progress(self, accuracy: AccuracyAuditResponse, recommended_arm: str | None, deploy_candidate: str | None) -> int:
        progress = 0
        if accuracy.detection_ready:
            progress += 35
        if accuracy.dataset_train_images > 0:
            progress += 12
        if accuracy.dataset_val_images > 0:
            progress += 10
        if accuracy.dataset_test_images > 0:
            progress += 10
        if accuracy.baseline_metric_summary:
            progress += 15
        if recommended_arm:
            progress += 10
        if deploy_candidate:
            progress += 8
        return min(100, progress)

    def _autolabel_progress(
        self,
        *,
        accuracy: AccuracyAuditResponse,
        readiness_phase_status: dict[str, str],
        evidence_recommended_arm: str | None,
        latest_metric_summary: str | None,
        autolabel_ready: bool,
        review_queue_ready: bool,
    ) -> tuple[int, list[str]]:
        progress = 0
        blockers: list[str] = []
        if accuracy.detection_ready:
            progress += 15
        if readiness_phase_status.get("phase2_retinex") == "done":
            progress += 8
        if readiness_phase_status.get("phase3_registration") == "done":
            progress += 12
        if readiness_phase_status.get("phase4_mertens") == "done":
            progress += 8
        if readiness_phase_status.get("phase5_daf") == "done":
            progress += 8
        if evidence_recommended_arm:
            progress += 8
        if latest_metric_summary:
            progress += 8
        if accuracy.dataset_train_images >= 20:
            progress += 8
        if accuracy.segmentation_ready:
            progress += 10
        if autolabel_ready:
            progress += 18
        else:
            blockers.append("Auto-label output dataset and provider path are not connected yet.")
            progress = min(progress, 45)
        if review_queue_ready:
            progress += 10
        else:
            blockers.append("Review queue has not been built from the auto-label proposals yet.")
        if not accuracy.segmentation_ready:
            blockers.append("Mask or polygon labels are still missing, so bbox-to-mask auto-label cannot be validated yet.")
            progress = min(progress, 65)
        return min(100, progress), blockers

    def _segmentation_progress(
        self,
        *,
        accuracy: AccuracyAuditResponse,
        training_runs: TrainingRunHistoryResponse,
        autolabel_ready: bool,
    ) -> int:
        progress = 0
        if accuracy.detection_ready:
            progress += 10
        if accuracy.segmentation_ready:
            progress += 45
        if autolabel_ready:
            progress += 10
        if accuracy.segmentation_bootstrap_ready:
            progress += 25
        if accuracy.segmentation_bootstrap_mode == "sam_refined_mask":
            progress += 10
        if accuracy.segmentation_bootstrap_refined_items > 0:
            progress += 5
        if self._has_segmentation_metrics(training_runs):
            progress += 20
        if accuracy.current_label_mode in {"yolo_segmentation", "mask"}:
            progress += 15
        return min(100, progress)

    def _reporting_progress(
        self,
        *,
        scorecard_ready: bool,
        accuracy_ready: bool,
        comparison_ready: bool,
        paper_pack_ready: bool,
        pipeline_ready: bool,
        export_ready: bool,
    ) -> int:
        progress = 0
        progress += 20 if scorecard_ready else 0
        progress += 20 if accuracy_ready else 0
        progress += 20 if comparison_ready else 0
        progress += 20 if paper_pack_ready else 0
        progress += 10 if pipeline_ready else 0
        progress += 10 if export_ready else 0
        return min(100, progress)

    def _build_structure_items(
        self,
        *,
        workspace_path: Path,
        readiness_phase_status: dict[str, str],
        accuracy: AccuracyAuditResponse,
        scorecard,
        pipeline_ready: bool,
        export_ready: bool,
        paper_pack_ready: bool,
        packaging_ready: bool,
        package_build_ready: bool,
        detection_progress: int,
        autolabel_progress: int,
        segmentation_progress: int,
        autolabel_blockers: list[str],
        review_queue_ready: bool,
        commercial_snapshot: dict | None,
    ) -> list[ProgramStructureItem]:
        data_progress = 100 if readiness_phase_status.get("phase0_split_freeze") == "done" and readiness_phase_status.get("phase1_bootstrap") == "done" else 45
        pixel_progress = (
            (30 if readiness_phase_status.get("phase2_retinex") == "done" else 0)
            + (25 if readiness_phase_status.get("phase4_mertens") == "done" else 0)
            + (25 if readiness_phase_status.get("phase5_daf") == "done" else 0)
            + (20 if (workspace_path / "evaluations" / "evidence" / "report.json").exists() else 0)
        )
        reporting_progress = self._reporting_progress(
            scorecard_ready=Path(scorecard.report_json_path).exists(),
            accuracy_ready=(workspace_path / "evaluations" / "accuracy_audit" / "report.json").exists(),
            comparison_ready=(workspace_path / "evaluations" / "reporting" / "arm_comparison.json").exists(),
            paper_pack_ready=paper_pack_ready,
            pipeline_ready=pipeline_ready,
            export_ready=export_ready,
        )
        ui_progress = 40
        if pipeline_ready:
            ui_progress += 25
        if export_ready:
            ui_progress += 15
        if Path(scorecard.report_json_path).exists():
            ui_progress += 10

        segmentation_blockers = []
        if not accuracy.segmentation_ready:
            segmentation_blockers.append("Current workspace is bbox-only, so true segmentation metrics are blocked.")
        if accuracy.segmentation_bootstrap_ready and not accuracy.segmentation_ready:
            segmentation_blockers.append("Coarse mask bootstrap exists, but a reviewed gold subset is still required for trustworthy segmentation claims.")
        if accuracy.segmentation_bootstrap_mode == "sam_refined_mask" and not accuracy.segmentation_ready:
            segmentation_blockers.append("SAM-refined masks reduce bootstrap noise, but they still require review and gold-mask promotion before mIoU or Dice claims.")

        items = [
            ProgramStructureItem(
                module_name="Data Freeze And Workspace Bootstrap",
                category="foundation",
                status=self._status_from_progress(data_progress),
                progress_percent=data_progress,
                summary="Lux candidate discovery, staging, pair manifests, and frozen group-level splits are already working.",
                blockers=[],
                next_step="Apply the same bootstrap loop to the larger ship-defect dataset.",
            ),
            ProgramStructureItem(
                module_name="Pixel Intelligence And Exposure Fusion",
                category="vision-core",
                status=self._status_from_progress(pixel_progress),
                progress_percent=min(100, pixel_progress),
                summary="Retinex, MergeMertens, and DAF branches are wired into the same workspace and evidence loop.",
                blockers=[],
                next_step="Add local relighting and target-lux region control after the current benchmark loop is stabilized.",
            ),
            ProgramStructureItem(
                module_name="Detection Benchmark Loop",
                category="evaluation",
                status=self._status_from_progress(detection_progress),
                progress_percent=detection_progress,
                summary="Detection-first benchmarking can run now because bbox labels, frozen splits, and training plans already exist.",
                blockers=[item for item in accuracy.blockers if "Segmentation" not in item][:3],
                next_step="Connect the real trainer command and collect raw160 / retinex / mertens / DAF metrics on the same split.",
            ),
            ProgramStructureItem(
                module_name="Auto-Label Loop",
                category="autolabel",
                status=self._status_from_progress(autolabel_progress),
                progress_percent=autolabel_progress,
                summary="Bootstrap proposals exist, and review queue integration determines whether the auto-label loop is actually operational.",
                blockers=autolabel_blockers,
                next_step="Use the review queue to validate proposals, then connect retraining so reviewed items feed the next detector iteration.",
            ),
            ProgramStructureItem(
                module_name="Segmentation Loop",
                category="segmentation",
                status="blocked" if not accuracy.segmentation_ready else self._status_from_progress(segmentation_progress),
                progress_percent=segmentation_progress,
                summary=(
                    "Segmentation work now has a SAM-refined bootstrap path, but it still needs review-grade masks before product or paper claims."
                    if accuracy.segmentation_bootstrap_mode == "sam_refined_mask"
                    else "Segmentation work now has a coarse bootstrap path, but it still needs review-grade masks before product or paper claims."
                ),
                blockers=segmentation_blockers,
                next_step=(
                    "Review the SAM-refined bootstrap masks, promote a gold subset, and then run segment training with mIoU and Dice reporting."
                    if accuracy.segmentation_bootstrap_mode == "sam_refined_mask"
                    else "Review the coarse bootstrap masks or add a SAM-style bootstrap path, then create a small gold set before claiming segmentation accuracy."
                ),
            ),
            ProgramStructureItem(
                module_name="Reporting, Paper Pack, And Exports",
                category="research-output",
                status=self._status_from_progress(reporting_progress),
                progress_percent=reporting_progress,
                summary="Scorecards, accuracy audit, arm comparison, paper pack, and Excel-compatible CSV exports are already connected.",
                blockers=[] if export_ready else ["CSV export bundle has not been generated yet."],
                next_step="Replace proxy evidence rows with real training metrics so the paper pack becomes publication-grade.",
            ),
            ProgramStructureItem(
                module_name="UI And Visual Traceability",
                category="experience",
                status=self._status_from_progress(min(100, ui_progress)),
                progress_percent=min(100, ui_progress),
                summary="The static UI already shows visual pipeline progress, artifact previews, live logs, scorecards, and research reports.",
                blockers=[],
                next_step="Keep the current static UI as the operator view, then migrate to React only after the core loop settles.",
            ),
            ProgramStructureItem(
                module_name="Desktop Packaging",
                category="distribution",
                status="implemented" if package_build_ready else ("partial" if packaging_ready else "planned"),
                progress_percent=100 if package_build_ready else (65 if packaging_ready else 25),
                summary="Windows exe packaging now has a desktop entry script, PyInstaller spec, and PowerShell build script.",
                blockers=[] if package_build_ready else ["Install PyInstaller and run the EXE build once in the packaging environment."],
                next_step="Smoke-test the packaged exe on a clean Windows machine and add application icon metadata.",
            ),
            ProgramStructureItem(
                module_name="Review Queue",
                category="human-in-the-loop",
                status="implemented" if review_queue_ready else "planned",
                progress_percent=100 if review_queue_ready else 35,
                summary="Review queue converts auto-label proposals into an explicit human validation workload instead of leaving them as raw files.",
                blockers=[] if review_queue_ready else ["Build the review queue from the auto-label bootstrap proposals."],
                next_step="Add approve/reject/edit actions so reviewed proposals can be fed back into retraining.",
            ),
        ]
        if commercial_snapshot is not None:
            items.append(
                ProgramStructureItem(
                    module_name="Commercialization Readiness",
                    category="commercial",
                    status=self._status_from_progress(int(commercial_snapshot.get("commercial_readiness_score", 0))),
                    progress_percent=int(commercial_snapshot.get("commercial_readiness_score", 0)),
                    summary=f"Protected-source intake and staged workspace expansion are tracked under commercial stage `{commercial_snapshot.get('commercial_stage', 'n/a')}`.",
                    blockers=list(commercial_snapshot.get("risks", []))[:3],
                    next_step=(commercial_snapshot.get("next_actions") or ["Stage more protected datasets and replace proxy metrics with real trainer runs."])[0],
                )
            )
        return items

    def _current_stage(self, *, detection_ready: bool, autolabel_progress: int, segmentation_ready: bool, field_score: int) -> str:
        if not detection_ready:
            return "Seed labeling and detection baseline bootstrap"
        if autolabel_progress < 45:
            return "Detection-first validation and auto-label foundation"
        if segmentation_ready:
            return "Field validation with real trainer integration" if field_score < 70 else "Stabilization and paper-grade benchmarking"
        if not segmentation_ready:
            return "Segmentation bootstrap and review"
        if field_score < 70:
            return "Field validation with real trainer integration"
        return "Stabilization and paper-grade benchmarking"

    def _next_actions(self, *, accuracy: AccuracyAuditResponse, scorecard, autolabel_progress: int, segmentation_progress: int, blockers: list[str]) -> list[str]:
        actions: list[str] = []
        if any("PyInstaller" in item for item in blockers):
            actions.append("Install PyInstaller, run the EXE build script once, and verify the packaged desktop launcher on Windows.")
        if any("ultralytics" in item for item in blockers):
            actions.append("Install or connect ultralytics so real raw160 / retinex / mertens / DAF accuracy runs can execute.")
        if any("opencv-python" in item for item in blockers):
            actions.append("Install opencv-python and switch the registration gate to the stronger ECC or ORB aligner.")
        if any("Review queue" in item for item in blockers):
            actions.append("Build the review queue and start checking high-priority val/test proposals before retraining.")
        if autolabel_progress < 60:
            actions.append("Implement detector-driven auto-label proposals and the review queue before scaling to the 6,000-image ship dataset.")
        if segmentation_progress < 50:
            actions.append("Create a small mask gold set or connect a SAM-style mask bootstrap path so segmentation metrics become measurable.")
        elif segmentation_progress < 75:
            actions.append("Review the segmentation bootstrap set and promote a gold subset before reporting mIoU or Dice.")
        if accuracy.dataset_val_images == 0 or accuracy.dataset_test_images == 0:
            actions.append("Backfill validation and test splits on transformed arms before making field or paper claims.")
        actions.extend(scorecard.next_actions[:2])
        return self._dedupe(actions)[:5]

    def _has_segmentation_metrics(self, training_runs: TrainingRunHistoryResponse) -> bool:
        metric_names = {"mIoU", "Dice", "boundary F1", "boundary_f1"}
        return any(any(name in run.metrics for name in metric_names) for run in training_runs.runs)

    def _status_from_progress(self, progress: int) -> str:
        if progress >= 85:
            return "implemented"
        if progress >= 55:
            return "partial"
        if progress >= 25:
            return "planned"
        return "blocked"

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _load_commercial_snapshot(self, workspace_path: Path) -> dict | None:
        report_path = workspace_path / "evaluations" / "commercialization" / "commercial_plan.json"
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_report(self, report: ProgramStatusResponse) -> None:
        report_json_path = Path(report.report_json_path)
        atomic_write_text(report_json_path, json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        lines = [
            "# Program Status",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Overall progress: **{report.overall_progress_percent}%**",
            f"- Execution readiness: **{report.execution_readiness_percent}%**",
            f"- Auto-label progress: **{report.autolabel_progress_percent}%**",
            f"- Detection progress: **{report.detection_progress_percent}%**",
            f"- Segmentation progress: **{report.segmentation_progress_percent}%**",
            f"- Current stage: **{report.current_stage}**",
            f"- Recommended arm: **{report.recommended_arm or 'n/a'}**",
            f"- Deploy candidate: **{report.deploy_candidate or 'n/a'}**",
            f"- Commercial stage: **{report.commercial_stage or 'n/a'}**",
            f"- Commercial readiness: **{report.commercial_readiness_score if report.commercial_readiness_score is not None else 'n/a'}**",
            "",
            "## Summary",
        ]
        for item in report.summary_points:
            lines.append(f"- {item}")
        lines.extend(["", "## Structure"])
        for item in report.structure_items:
            lines.append(
                f"- `{item.module_name}` [{item.category}] status={item.status}, progress={item.progress_percent}%"
            )
            lines.append(f"  - summary: {item.summary}")
            if item.blockers:
                lines.append(f"  - blockers: {' | '.join(item.blockers)}")
            lines.append(f"  - next: {item.next_step}")
        lines.extend(["", "## Blockers"])
        for item in report.blockers:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in report.next_actions:
            lines.append(f"- {item}")
        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n")


program_status_service = ProgramStatusService()
