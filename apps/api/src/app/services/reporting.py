from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    ArmComparisonResponse,
    ArmComparisonRow,
    CsvExportResponse,
    EvaluationRunRequest,
    PaperAblationRow,
    PaperPackResponse,
    UsageScorecardResponse,
)
from app.services.accuracy_audit import accuracy_audit_service
from app.services.benchmark import benchmark_service
from app.services.data_quality_audit import data_quality_audit_service
from app.services.desktop_package import desktop_package_service
from app.services.evaluation import evaluation_service
from app.services.review_queue import review_queue_service
from app.services.training import training_service


class ReportingService:
    def build_scorecard(self, workspace_root: str) -> UsageScorecardResponse:
        workspace_path = Path(workspace_root)
        # Scorecard should reflect the current runtime, not whichever readiness file
        # happened to be written before dependencies were installed.
        readiness = evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=str(workspace_path),
                include_arms=list(evaluation_service.ARM_ROOTS.keys()),
                refresh_report=True,
            )
        )
        evidence = benchmark_service.load_report(workspace_path)
        training_runs = self._safe_training_runs(workspace_path)
        pipeline_report_path = workspace_path / "evaluations" / "pipeline" / "report.json"
        package_plan = self._safe_package_plan(workspace_path)
        review_queue = self._safe_review_queue(workspace_path)
        data_quality_audit = self._safe_data_quality_audit(workspace_path)

        export_root = workspace_path / "evaluations" / "exports"
        export_ready = export_root.exists() and any(export_root.glob("*.csv"))
        visual_trace_ready = pipeline_report_path.exists()
        review_queue_ready = review_queue is not None
        reviewed_items = 0 if review_queue is None else sum(
            count for status, count in review_queue.status_counts.items() if status != "pending"
        )

        required_programs = [item.name for item in readiness.requirements if item.category == "program" and item.required]
        validation_gap_count = sum(
            1
            for arm in readiness.arms
            if any("Validation split is empty" in note or "Test split is empty" in note for note in arm.notes)
        )
        has_completed_live_run = any(run.status == "completed" and not run.dry_run for run in training_runs.runs)
        latest_run = training_runs.runs[0] if training_runs.runs else None
        latest_metric_summary = self._latest_metric_summary(latest_run.metrics if latest_run else {})

        research_score = min(
            100,
            round(
                (readiness.completion_percent * 0.42)
                + (readiness.execution_readiness_percent * 0.28)
                + (12 if evidence.recommended_arm else 0)
                + (8 if latest_run else 0)
                + (10 if visual_trace_ready else 0)
                + (4 if review_queue_ready else 0)
            ),
        )
        field_score = max(
            0,
            min(
                100,
                round(
                    (readiness.completion_percent * 0.18)
                    + (readiness.execution_readiness_percent * 0.46)
                    + (10 if evidence.recommended_arm else 0)
                    + (8 if has_completed_live_run else 0)
                    + (6 if export_ready else 0)
                    + (6 if review_queue_ready else 0)
                    + (6 if package_plan and package_plan.build_ready else 0)
                    - (validation_gap_count * 4)
                    - (len(required_programs) * 6)
                ),
            ),
        )
        production_score = max(
            0,
            min(
                100,
                round(
                    (readiness.completion_percent * 0.15)
                    + (readiness.execution_readiness_percent * 0.32)
                    + (8 if export_ready else 0)
                    + (8 if has_completed_live_run else 0)
                    + (4 if latest_metric_summary else 0)
                    + (8 if review_queue_ready else 0)
                    + (16 if package_plan and package_plan.build_ready else 0)
                    - (validation_gap_count * 6)
                    - (len(required_programs) * 8)
                ),
            ),
        )

        strengths = [
            f"Pipeline structure completion is {readiness.completion_percent}%.",
            f"Execution readiness is {readiness.execution_readiness_percent}%, and the current recommended arm is {evidence.recommended_arm or 'n/a'}.",
            "Visual pipeline progress, logs, and artifact previews are already available in the UI." if visual_trace_ready else "Visual pipeline reporting has not been generated yet.",
            "Desktop EXE packaging is build-ready." if package_plan and package_plan.build_ready else "Desktop EXE packaging is still in planning mode.",
            f"Review queue is active with {reviewed_items} reviewed items." if review_queue_ready else "Review queue has not been built yet.",
            (
                f"Data quality audit found {data_quality_audit.position_issue_count} position issues and {data_quality_audit.label_issue_count} label issues."
                if data_quality_audit is not None
                else "Data quality audit has not been generated yet."
            ),
        ]
        if latest_metric_summary:
            strengths.append(f"Latest training metric summary: {latest_metric_summary}.")
        blockers = []
        if required_programs:
            blockers.append(f"Missing required programs: {', '.join(required_programs)}")
        if validation_gap_count:
            blockers.append(f"Validation/test split warnings remain in {validation_gap_count} arms.")
        if not export_ready:
            blockers.append("CSV export bundle has not been generated yet.")
        if not has_completed_live_run:
            blockers.append("Not enough completed live training runs have been collected yet.")
        if not review_queue_ready:
            blockers.append("Review queue has not been built from the auto-label proposals yet.")
        if data_quality_audit is not None and data_quality_audit.severe_position_issue_count:
            blockers.append(f"Severe photo misalignment remains in {data_quality_audit.severe_position_issue_count} groups.")
        if data_quality_audit is not None and data_quality_audit.label_issue_count:
            blockers.append(f"Label audit found {data_quality_audit.label_issue_count} suspect label files.")
        if package_plan and not package_plan.build_ready:
            blockers.extend(package_plan.blockers)

        next_actions = [
            "Connect ultralytics or a real external trainer command to the segmentation training environment.",
            "Backfill validation/test splits so field usability confidence improves.",
            "Store training result files (results.csv/results.json) so arm comparisons stay populated.",
            "Run the same loop on the 6,000-image ship-defect dataset to collect real field metrics.",
        ]
        if not review_queue_ready:
            next_actions.insert(0, "Build the review queue and start reviewing high-priority val/test proposals.")
        if package_plan and not package_plan.build_ready:
            next_actions.insert(0, "Install PyInstaller and run the desktop build script so the operator UI can ship as an exe.")
        if data_quality_audit is not None and (data_quality_audit.position_issue_count or data_quality_audit.label_issue_count):
            next_actions.insert(0, "Use the data quality audit to fix misaligned groups and suspect labels before final accuracy claims.")

        reporting_root = workspace_path / "evaluations" / "reporting"
        reporting_root.mkdir(parents=True, exist_ok=True)
        response = UsageScorecardResponse(
            workspace_root=str(workspace_path),
            research_score=research_score,
            field_score=field_score,
            production_score=production_score,
            export_ready=export_ready,
            visual_trace_ready=visual_trace_ready,
            recommended_arm=evidence.recommended_arm,
            latest_metric_summary=latest_metric_summary,
            strengths=strengths,
            blockers=blockers,
            next_actions=next_actions,
            report_json_path=str(reporting_root / "scorecard.json"),
            report_markdown_path=str(reporting_root / "scorecard.md"),
        )
        self._write_scorecard(response)
        return response

    def load_scorecard(self, workspace_root: str) -> UsageScorecardResponse:
        path = Path(workspace_root) / "evaluations" / "reporting" / "scorecard.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing scorecard: {path}")
        return UsageScorecardResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def export_csv_bundle(self, workspace_root: str) -> CsvExportResponse:
        workspace_path = Path(workspace_root)
        readiness = evaluation_service.load_readiness_report(workspace_path)
        evidence = benchmark_service.load_report(workspace_path)
        training_runs = self._safe_training_runs(workspace_path)
        comparison = self.build_arm_comparison(workspace_root)
        paper_pack = self.build_paper_pack(workspace_root)
        accuracy_audit = accuracy_audit_service.build_report(workspace_root)
        data_quality_audit = data_quality_audit_service.build_report(workspace_root)

        export_root = workspace_path / "evaluations" / "exports"
        export_root.mkdir(parents=True, exist_ok=True)

        scorecard = self.build_scorecard(workspace_root)
        scorecard_csv = export_root / "scorecard.csv"
        readiness_csv = export_root / "readiness_arms.csv"
        evidence_csv = export_root / "evidence_arms.csv"
        training_csv = export_root / "training_runs.csv"
        comparison_csv = export_root / "arm_comparison.csv"
        paper_summary_csv = export_root / "paper_pack_summary.csv"
        accuracy_audit_csv = export_root / "accuracy_audit.csv"
        data_quality_audit_csv = export_root / "data_quality_audit.csv"
        segmentation_bootstrap_csv = export_root / "segmentation_bootstrap.csv"

        self._write_csv(
            scorecard_csv,
            ["research_score", "field_score", "production_score", "recommended_arm", "latest_metric_summary", "export_ready", "visual_trace_ready"],
            [[
                scorecard.research_score,
                scorecard.field_score,
                scorecard.production_score,
                scorecard.recommended_arm or "",
                scorecard.latest_metric_summary or "",
                scorecard.export_ready,
                scorecard.visual_trace_ready,
            ]],
        )
        self._write_csv(
            readiness_csv,
            ["arm_name", "ready", "train_images", "val_images", "test_images", "dataset_yaml_path", "notes"],
            [
                [
                    arm.arm_name,
                    arm.ready,
                    arm.split_image_counts.get("train", 0),
                    arm.split_image_counts.get("val", 0),
                    arm.split_image_counts.get("test", 0),
                    arm.dataset_yaml_path or "",
                    " | ".join(arm.notes),
                ]
                for arm in readiness.arms
            ],
        )
        self._write_csv(
            evidence_csv,
            [
                "arm_name",
                "available_groups",
                "avg_dynamic_range",
                "avg_defect_visibility",
                "avg_background_suppression",
                "coverage_ratio",
                "evidence_score",
                "visibility_gain_vs_raw80",
                "dynamic_range_gain_vs_raw80",
            ],
            [
                [
                    arm.arm_name,
                    arm.available_groups,
                    arm.avg_dynamic_range,
                    arm.avg_defect_visibility,
                    arm.avg_background_suppression,
                    arm.coverage_ratio,
                    arm.evidence_score,
                    arm.visibility_gain_vs_raw80,
                    arm.dynamic_range_gain_vs_raw80,
                ]
                for arm in evidence.arms
            ],
        )
        self._write_csv(
            training_csv,
            [
                "run_name",
                "arm",
                "status",
                "dry_run",
                "started_at",
                "finished_at",
                "return_code",
                "primary_metric",
                "output_dir",
            ],
            [
                [
                    run.run_name,
                    run.arm,
                    run.status,
                    run.dry_run,
                    run.started_at or "",
                    run.finished_at or "",
                    run.return_code if run.return_code is not None else "",
                    self._latest_metric_summary(run.metrics),
                    run.output_dir,
                ]
                for run in training_runs.runs
            ],
        )
        self._write_csv(
            comparison_csv,
            [
                "arm_name",
                "ready",
                "available_groups",
                "coverage_ratio",
                "evidence_score",
                "visibility_gain_vs_raw80",
                "dynamic_range_gain_vs_raw80",
                "latest_metric_name",
                "latest_metric_value",
                "latest_run_name",
                "status",
                "decision_tag",
                "notes",
            ],
            [
                [
                    row.arm_name,
                    row.ready,
                    row.available_groups,
                    row.coverage_ratio,
                    row.evidence_score,
                    row.visibility_gain_vs_raw80,
                    row.dynamic_range_gain_vs_raw80,
                    row.latest_metric_name or "",
                    row.latest_metric_value if row.latest_metric_value is not None else "",
                    row.latest_run_name or "",
                    row.status,
                    row.decision_tag,
                    " | ".join(row.notes),
                ]
                for row in comparison.rows
            ],
        )
        self._write_csv(
            paper_summary_csv,
            [
                "paper_readiness_score",
                "impact_domain",
                "working_title",
                "novelty_statement",
                "target_problem",
                "report_json_path",
                "report_markdown_path",
                "ablation_csv_path",
            ],
            [[
                paper_pack.paper_readiness_score,
                paper_pack.impact_domain,
                paper_pack.working_title,
                paper_pack.novelty_statement,
                paper_pack.target_problem,
                paper_pack.report_json_path,
                paper_pack.report_markdown_path,
                paper_pack.ablation_csv_path,
            ]],
        )
        self._write_csv(
            accuracy_audit_csv,
            [
                "accuracy_readiness_score",
                "current_label_mode",
                "detection_ready",
                "segmentation_ready",
                "segmentation_bootstrap_ready",
                "segmentation_bootstrap_dataset_root",
                "segmentation_bootstrap_mode",
                "segmentation_bootstrap_refined_items",
                "dataset_train_images",
                "dataset_val_images",
                "dataset_test_images",
                "baseline_arm",
                "baseline_metric_summary",
                "report_json_path",
                "report_markdown_path",
            ],
            [[
                accuracy_audit.accuracy_readiness_score,
                accuracy_audit.current_label_mode,
                accuracy_audit.detection_ready,
                accuracy_audit.segmentation_ready,
                accuracy_audit.segmentation_bootstrap_ready,
                accuracy_audit.segmentation_bootstrap_dataset_root or "",
                accuracy_audit.segmentation_bootstrap_mode or "",
                accuracy_audit.segmentation_bootstrap_refined_items,
                accuracy_audit.dataset_train_images,
                accuracy_audit.dataset_val_images,
                accuracy_audit.dataset_test_images,
                accuracy_audit.baseline_arm or "",
                accuracy_audit.baseline_metric_summary or "",
                accuracy_audit.report_json_path,
                accuracy_audit.report_markdown_path,
            ]],
        )
        self._write_csv(
            data_quality_audit_csv,
            [
                "registration_reports_scanned",
                "registered_groups_scanned",
                "position_issue_count",
                "severe_position_issue_count",
                "label_files_scanned",
                "label_issue_count",
                "invalid_label_count",
                "out_of_bounds_box_count",
                "tiny_box_count",
                "oversize_box_count",
                "duplicate_box_count",
                "report_json_path",
                "report_markdown_path",
            ],
            [[
                data_quality_audit.registration_reports_scanned,
                data_quality_audit.registered_groups_scanned,
                data_quality_audit.position_issue_count,
                data_quality_audit.severe_position_issue_count,
                data_quality_audit.label_files_scanned,
                data_quality_audit.label_issue_count,
                data_quality_audit.invalid_label_count,
                data_quality_audit.out_of_bounds_box_count,
                data_quality_audit.tiny_box_count,
                data_quality_audit.oversize_box_count,
                data_quality_audit.duplicate_box_count,
                data_quality_audit.report_json_path,
                data_quality_audit.report_markdown_path,
            ]],
        )
        segmentation_report_path = workspace_path / "evaluations" / "segmentation_bootstrap" / "report.json"
        if segmentation_report_path.exists():
            segmentation_payload = json.loads(segmentation_report_path.read_text(encoding="utf-8"))
            self._write_csv(
                segmentation_bootstrap_csv,
                [
                    "source_dataset_name",
                    "dataset_root",
                    "mask_root",
                    "total_items",
                    "review_required_count",
                    "train_count",
                    "val_count",
                    "test_count",
                    "bootstrap_mode",
                    "sam_used",
                    "sam_model",
                    "sam_device",
                    "refined_items",
                    "report_json_path",
                ],
                [[
                    segmentation_payload.get("source_dataset_name", ""),
                    segmentation_payload.get("dataset_root", ""),
                    segmentation_payload.get("mask_root", ""),
                    segmentation_payload.get("total_items", 0),
                    segmentation_payload.get("review_required_count", 0),
                    (segmentation_payload.get("split_counts") or {}).get("train", 0),
                    (segmentation_payload.get("split_counts") or {}).get("val", 0),
                    (segmentation_payload.get("split_counts") or {}).get("test", 0),
                    segmentation_payload.get("bootstrap_mode", ""),
                    segmentation_payload.get("sam_used", False),
                    segmentation_payload.get("sam_model", ""),
                    segmentation_payload.get("sam_device", ""),
                    segmentation_payload.get("refined_items", 0),
                    segmentation_payload.get("report_json_path", ""),
                ]],
            )
        scorecard = self.build_scorecard(workspace_root)

        return CsvExportResponse(
            workspace_root=str(workspace_path),
            export_root=str(export_root),
            files={
                "scorecard_csv": str(scorecard_csv),
                "readiness_csv": str(readiness_csv),
                "evidence_csv": str(evidence_csv),
                "training_runs_csv": str(training_csv),
                "arm_comparison_csv": str(comparison_csv),
                "paper_pack_summary_csv": str(paper_summary_csv),
                "paper_pack_ablation_csv": paper_pack.ablation_csv_path,
                "accuracy_audit_csv": str(accuracy_audit_csv),
                "data_quality_audit_csv": str(data_quality_audit_csv),
                **({"segmentation_bootstrap_csv": str(segmentation_bootstrap_csv)} if segmentation_report_path.exists() else {}),
            },
            message="CSV export bundle created. These files are Excel-compatible.",
        )

    def build_arm_comparison(self, workspace_root: str) -> ArmComparisonResponse:
        workspace_path = Path(workspace_root)
        readiness = evaluation_service.load_readiness_report(workspace_path)
        evidence = benchmark_service.load_report(workspace_path)
        training_runs = self._safe_training_runs(workspace_path)
        readiness_map = {arm.arm_name: arm for arm in readiness.arms}
        evidence_map = {arm.arm_name: arm for arm in evidence.arms}
        latest_by_arm = {}
        for run in training_runs.runs:
            if run.arm not in latest_by_arm:
                latest_by_arm[run.arm] = run

        rows: list[ArmComparisonRow] = []
        for arm_name in ["raw160", "retinex", "mertens", "daf", "mixed_raw_forensic_wdr"]:
            ready_arm = readiness_map.get(arm_name)
            evidence_arm = evidence_map.get("retinex80" if arm_name == "retinex" else arm_name)
            latest_run = latest_by_arm.get(arm_name)
            metric_name, metric_value = self._pick_primary_metric(latest_run.metrics if latest_run else {})
            decision_tag = "review"
            notes = []
            if evidence_arm and evidence_arm.arm_name in {evidence.recommended_arm, evidence.peak_arm}:
                decision_tag = "promising"
            if latest_run and latest_run.status == "completed" and metric_name is not None:
                decision_tag = "deploy-check"
                notes.append(f"Latest training metric {metric_name}={metric_value}.")
            if ready_arm and ready_arm.notes:
                notes.extend(ready_arm.notes[:2])
            rows.append(
                ArmComparisonRow(
                    arm_name=arm_name,
                    ready=bool(ready_arm.ready) if ready_arm else False,
                    available_groups=evidence_arm.available_groups if evidence_arm else 0,
                    coverage_ratio=evidence_arm.coverage_ratio if evidence_arm else None,
                    evidence_score=evidence_arm.evidence_score if evidence_arm else None,
                    visibility_gain_vs_raw80=evidence_arm.visibility_gain_vs_raw80 if evidence_arm else None,
                    dynamic_range_gain_vs_raw80=evidence_arm.dynamic_range_gain_vs_raw80 if evidence_arm else None,
                    latest_metric_name=metric_name,
                    latest_metric_value=metric_value,
                    latest_run_name=latest_run.run_name if latest_run else None,
                    status=latest_run.status if latest_run else ("ready" if ready_arm and ready_arm.ready else "pending"),
                    decision_tag=decision_tag,
                    notes=notes,
                )
            )

        evidence_candidate = evidence.recommended_arm
        training_leader = self._best_training_arm(rows)
        deploy_candidate = training_leader or evidence_candidate
        findings = [
            f"Evidence leader: {evidence_candidate or 'n/a'}",
            f"Training leader: {training_leader or 'n/a'}",
            f"Current deploy candidate: {deploy_candidate or 'n/a'}",
        ]
        if evidence_candidate == "retinex80":
            findings.append("Retinex remains the safest current branch because it leads evidence score with full coverage.")
        comparison_root = workspace_path / "evaluations" / "reporting"
        comparison_root.mkdir(parents=True, exist_ok=True)
        response = ArmComparisonResponse(
            workspace_root=str(workspace_path),
            deploy_candidate=deploy_candidate,
            evidence_candidate=evidence_candidate,
            training_candidate=training_leader,
            rows=rows,
            key_findings=findings,
            report_json_path=str(comparison_root / "arm_comparison.json"),
            report_markdown_path=str(comparison_root / "arm_comparison.md"),
        )
        self._write_arm_comparison(response)
        return response

    def load_arm_comparison(self, workspace_root: str) -> ArmComparisonResponse:
        path = Path(workspace_root) / "evaluations" / "reporting" / "arm_comparison.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing arm comparison: {path}")
        return ArmComparisonResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def build_paper_pack(self, workspace_root: str) -> PaperPackResponse:
        workspace_path = Path(workspace_root)
        scorecard = self.build_scorecard(workspace_root)
        comparison = self.build_arm_comparison(workspace_root)
        readiness = evaluation_service.load_readiness_report(workspace_path)
        evidence = benchmark_service.load_report(workspace_path)
        summary = self._load_dataset_summary(workspace_path)
        domain = self._infer_impact_domain(summary.get("dataset_path", ""), str(workspace_path))
        latest_metric = scorecard.latest_metric_summary or "no training metric yet"

        title_candidates = [
            f"Defect-Aware Multi-Exposure Fusion for {domain}",
            f"From Low-Lux Inputs to Segmentation-Ready Views in {domain}",
            f"Retinex, Exposure Fusion, and Defect Priors for {domain}",
        ]
        working_title = title_candidates[0]
        novelty_statement = (
            "This pipeline uses only existing multi-exposure inspection datasets and turns them into "
            "a reproducible comparison between raw inputs, Retinex restoration, MergeMertens fusion, "
            "and defect-aware fusion without collecting new labels."
        )
        target_problem = (
            f"Improve low-light and mixed-exposure defect segmentation in {domain.lower()} while keeping "
            "label reuse, visual traceability, and training readiness inside one workflow."
        )

        visibility_arm = comparison.evidence_candidate or evidence.recommended_arm or "retinex80"
        deploy_arm = comparison.deploy_candidate or comparison.training_candidate or "raw160"
        validation_gap_count = sum(
            1
            for arm in readiness.arms
            if any("Validation split is empty" in note or "Test split is empty" in note for note in arm.notes)
        )
        live_metric_bonus = 8 if scorecard.latest_metric_summary else 0
        paper_readiness_score = max(
            0,
            min(
                100,
                round(
                    (scorecard.research_score * 0.55)
                    + (readiness.execution_readiness_percent * 0.2)
                    + (10 if evidence.recommended_arm else 0)
                    + live_metric_bonus
                    - (validation_gap_count * 4)
                ),
            ),
        )

        abstract_draft = (
            f"We present a practical research platform for {domain.lower()} that converts existing multi-exposure "
            "inspection images into segmentation-ready training assets. The system freezes group-based splits, "
            "builds restored and fused variants, verifies registration before label reuse, and compares raw, "
            "Retinex, MergeMertens, and defect-aware fusion branches under one reproducible workflow. In the current "
            f"workspace, the evidence leader is {visibility_arm}, the current deploy candidate is {deploy_arm}, "
            f"and the latest recorded training metric is {latest_metric}. The resulting platform is already strong "
            "enough for research iteration and supports ablation tables, visual traces, Excel exports, and live "
            "training logs from the same UI."
        )

        contributions = [
            "A defect research workspace that keeps enhancement, registration, fusion, labeling evidence, training plans, and exports in one reproducible loop.",
            "A defect-aware exposure-fusion branch that can be compared directly against raw, Retinex, and MergeMertens arms using the same frozen split.",
            "A visual execution trace that stores stage-by-stage previews, logs, and artifact paths so qualitative evidence is preserved for papers and reviews.",
            "A paper-oriented reporting layer that turns existing datasets into scorecards, ablation rows, comparison tables, and exportable research artifacts.",
        ]
        experiment_protocol = [
            "Freeze train/val/test by group_id so exposure variants from the same scene never leak across splits.",
            "Reuse only anchor labels that pass registration constraints before materializing transformed datasets.",
            "Compare raw160, retinex, mertens, and daf arms on identical splits and identical downstream training settings.",
            "Report both evidence proxies and downstream training metrics so visual improvements are tied back to defect performance.",
            "Export markdown plus CSV tables for figures, appendices, and spreadsheet review.",
        ]

        ablation_rows: list[PaperAblationRow] = []
        for row in comparison.rows:
            interpretation = self._interpret_arm(row)
            ablation_rows.append(
                PaperAblationRow(
                    arm_name=row.arm_name,
                    evidence_score=row.evidence_score,
                    visibility_gain_vs_raw80=row.visibility_gain_vs_raw80,
                    dynamic_range_gain_vs_raw80=row.dynamic_range_gain_vs_raw80,
                    latest_metric_name=row.latest_metric_name,
                    latest_metric_value=row.latest_metric_value,
                    decision_tag=row.decision_tag,
                    interpretation=interpretation,
                )
            )

        figure_checklist = [
            "Failure case grid: raw80 vs retinex80 vs mertens vs daf on the same defect group.",
            "Registration gate examples showing accepted and rejected label-reuse cases.",
            "Evidence benchmark table with coverage ratio, evidence score, and visibility gain.",
            "Training comparison table summarizing mAP, recall, or mIoU by arm.",
            "Pipeline UI screenshot showing visual progress, previews, and logs.",
        ]
        reproducibility_checklist = [
            f"Workspace root is fixed at {workspace_path}.",
            "Frozen split manifest exists and is group-based.",
            "Every report has JSON and markdown output paths saved to disk.",
            "Training runs store command previews, logs, status, and parsed metrics.",
            "CSV exports are Excel-compatible for external review.",
        ]
        limitations = list(scorecard.blockers)
        if not limitations:
            limitations.append("No explicit blockers are currently registered.")
        next_paper_actions = [
            "Run the same pipeline on the larger ship-defect dataset to strengthen external validity.",
            "Replace proxy evidence with real segmentation metrics from a connected trainer command.",
            "Collect failure cases where DAF improves visibility but not downstream metrics, then analyze why.",
            "Finalize an ablation section comparing raw, Retinex, MergeMertens, and defect-aware fusion under one trainer configuration.",
        ]

        paper_root = workspace_path / "evaluations" / "paper_pack"
        paper_root.mkdir(parents=True, exist_ok=True)
        response = PaperPackResponse(
            workspace_root=str(workspace_path),
            impact_domain=domain,
            paper_readiness_score=paper_readiness_score,
            working_title=working_title,
            title_candidates=title_candidates,
            novelty_statement=novelty_statement,
            target_problem=target_problem,
            abstract_draft=abstract_draft,
            contributions=contributions,
            experiment_protocol=experiment_protocol,
            ablation_rows=ablation_rows,
            figure_checklist=figure_checklist,
            reproducibility_checklist=reproducibility_checklist,
            limitations=limitations,
            next_paper_actions=next_paper_actions,
            report_json_path=str(paper_root / "paper_pack.json"),
            report_markdown_path=str(paper_root / "paper_pack.md"),
            ablation_csv_path=str(paper_root / "ablation_table.csv"),
        )
        self._write_paper_pack(response)
        return response

    def load_paper_pack(self, workspace_root: str) -> PaperPackResponse:
        path = Path(workspace_root) / "evaluations" / "paper_pack" / "paper_pack.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing paper pack: {path}")
        return PaperPackResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def _safe_training_runs(self, workspace_path: Path):
        try:
            return training_service.list_training_runs(str(workspace_path))
        except FileNotFoundError:
            from app.domain.research_models import TrainingRunHistoryResponse

            return TrainingRunHistoryResponse(workspace_root=str(workspace_path), total_runs=0, runs=[])

    def _safe_data_quality_audit(self, workspace_path: Path):
        try:
            return data_quality_audit_service.load_report(str(workspace_path))
        except FileNotFoundError:
            return None

    def _safe_package_plan(self, workspace_path: Path):
        return desktop_package_service.build_plan(str(workspace_path))

    def _safe_review_queue(self, workspace_path: Path):
        try:
            return review_queue_service.load_queue(str(workspace_path))
        except FileNotFoundError:
            return None

    def _latest_metric_summary(self, metrics: dict[str, float | int | str]) -> str | None:
        key, value = self._pick_primary_metric(metrics)
        if key is None:
            return None
        return f"{key}={value}"

    def _pick_primary_metric(self, metrics: dict[str, float | int | str]) -> tuple[str | None, float | int | str | None]:
        if not metrics:
            return None, None
        for key in [
            "mIoU",
            "Dice",
            "boundary F1",
            "boundary_f1",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/recall(B)",
            "metrics/precision(B)",
            "accuracy",
        ]:
            if key in metrics:
                return key, metrics[key]
        first_key = next(iter(metrics.keys()), None)
        if first_key is None:
            return None, None
        return first_key, metrics[first_key]

    def _best_training_arm(self, rows: list[ArmComparisonRow]) -> str | None:
        candidates = [row for row in rows if row.latest_metric_value is not None]
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda row: (
                self._metric_sort_value(row.latest_metric_value),
                row.evidence_score or 0.0,
            ),
            reverse=True,
        )
        return ranked[0].arm_name

    def _metric_sort_value(self, value: float | int | str | None) -> float:
        if value is None:
            return float("-inf")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    def _load_dataset_summary(self, workspace_path: Path) -> dict:
        summary_path = workspace_path / "manifests" / "summary.json"
        if not summary_path.exists():
            return {}
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _infer_impact_domain(self, dataset_path: str, workspace_root: str) -> str:
        source = f"{dataset_path} {workspace_root}".lower()
        if any(token in source for token in ["ship", "vessel", "marine", "선박"]):
            return "Maritime defect inspection"
        if any(token in source for token in ["paint", "coating", "도장"]):
            return "Coating and surface defect inspection"
        return "Industrial defect inspection"

    def _interpret_arm(self, row: ArmComparisonRow) -> str:
        if row.latest_metric_name and row.latest_metric_value is not None:
            return "Best current downstream training evidence."
        if row.decision_tag == "promising" and (row.visibility_gain_vs_raw80 or 0) > 0:
            return "Best current visual evidence branch for paper figures and defect visibility claims."
        if row.decision_tag == "review":
            return "Useful as an ablation baseline but needs stronger downstream validation."
        return "Needs more evidence before it can support the main paper claim."

    def _write_scorecard(self, scorecard: UsageScorecardResponse) -> None:
        json_path = Path(scorecard.report_json_path)
        atomic_write_text(json_path, json.dumps(scorecard.model_dump(), ensure_ascii=False, indent=2))
        lines = [
            "# Usage Scorecard",
            "",
            f"- Workspace: `{scorecard.workspace_root}`",
            f"- Research score: **{scorecard.research_score}**",
            f"- Field score: **{scorecard.field_score}**",
            f"- Production score: **{scorecard.production_score}**",
            f"- Recommended arm: **{scorecard.recommended_arm or 'n/a'}**",
            f"- Latest metric summary: **{scorecard.latest_metric_summary or 'n/a'}**",
            "",
            "## Strengths",
        ]
        for item in scorecard.strengths:
            lines.append(f"- {item}")
        lines.extend(["", "## Blockers"])
        for item in scorecard.blockers:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in scorecard.next_actions:
            lines.append(f"- {item}")
        atomic_write_text(scorecard.report_markdown_path, "\n".join(lines) + "\n")

    def _write_arm_comparison(self, comparison: ArmComparisonResponse) -> None:
        json_path = Path(comparison.report_json_path)
        atomic_write_text(json_path, json.dumps(comparison.model_dump(), ensure_ascii=False, indent=2))
        lines = [
            "# Arm Comparison",
            "",
            f"- Workspace: `{comparison.workspace_root}`",
            f"- Deploy candidate: `{comparison.deploy_candidate or 'n/a'}`",
            f"- Evidence candidate: `{comparison.evidence_candidate or 'n/a'}`",
            f"- Training candidate: `{comparison.training_candidate or 'n/a'}`",
            "",
            "## Findings",
        ]
        for item in comparison.key_findings:
            lines.append(f"- {item}")
        lines.extend(["", "## Rows"])
        for row in comparison.rows:
            lines.append(
                f"- `{row.arm_name}`: ready={row.ready}, evidence_score={row.evidence_score}, "
                f"metric={row.latest_metric_name or 'n/a'}={row.latest_metric_value if row.latest_metric_value is not None else 'n/a'}, "
                f"decision={row.decision_tag}, status={row.status}"
            )
        atomic_write_text(comparison.report_markdown_path, "\n".join(lines) + "\n")

    def _write_paper_pack(self, paper_pack: PaperPackResponse) -> None:
        json_path = Path(paper_pack.report_json_path)
        atomic_write_text(json_path, json.dumps(paper_pack.model_dump(), ensure_ascii=False, indent=2))
        self._write_csv(
            Path(paper_pack.ablation_csv_path),
            [
                "arm_name",
                "evidence_score",
                "visibility_gain_vs_raw80",
                "dynamic_range_gain_vs_raw80",
                "latest_metric_name",
                "latest_metric_value",
                "decision_tag",
                "interpretation",
            ],
            [
                [
                    row.arm_name,
                    row.evidence_score,
                    row.visibility_gain_vs_raw80,
                    row.dynamic_range_gain_vs_raw80,
                    row.latest_metric_name or "",
                    row.latest_metric_value if row.latest_metric_value is not None else "",
                    row.decision_tag,
                    row.interpretation,
                ]
                for row in paper_pack.ablation_rows
            ],
        )
        lines = [
            "# Paper Pack",
            "",
            f"- Workspace: `{paper_pack.workspace_root}`",
            f"- Impact domain: **{paper_pack.impact_domain}**",
            f"- Paper readiness score: **{paper_pack.paper_readiness_score}/100**",
            f"- Working title: **{paper_pack.working_title}**",
            "",
            "## Title Candidates",
        ]
        for item in paper_pack.title_candidates:
            lines.append(f"- {item}")
        lines.extend([
            "",
            "## Novelty Statement",
            paper_pack.novelty_statement,
            "",
            "## Target Problem",
            paper_pack.target_problem,
            "",
            "## Abstract Draft",
            paper_pack.abstract_draft,
            "",
            "## Contributions",
        ])
        for item in paper_pack.contributions:
            lines.append(f"- {item}")
        lines.extend(["", "## Experiment Protocol"])
        for item in paper_pack.experiment_protocol:
            lines.append(f"- {item}")
        lines.extend(["", "## Ablation Table"])
        for row in paper_pack.ablation_rows:
            lines.append(
                f"- `{row.arm_name}`: evidence={row.evidence_score}, visibility_gain={row.visibility_gain_vs_raw80}, "
                f"range_gain={row.dynamic_range_gain_vs_raw80}, metric={row.latest_metric_name or 'n/a'}="
                f"{row.latest_metric_value if row.latest_metric_value is not None else 'n/a'}, note={row.interpretation}"
            )
        lines.extend(["", "## Figure Checklist"])
        for item in paper_pack.figure_checklist:
            lines.append(f"- {item}")
        lines.extend(["", "## Reproducibility Checklist"])
        for item in paper_pack.reproducibility_checklist:
            lines.append(f"- {item}")
        lines.extend(["", "## Limitations"])
        for item in paper_pack.limitations:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Paper Actions"])
        for item in paper_pack.next_paper_actions:
            lines.append(f"- {item}")
        atomic_write_text(paper_pack.report_markdown_path, "\n".join(lines) + "\n")

    def _write_csv(self, path: Path, header: list[str], rows: list[list]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


reporting_service = ReportingService()
