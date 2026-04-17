from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    AccuracyAuditArm,
    AccuracyAuditResponse,
    EvaluationRunRequest,
    PixelMethodPlan,
    TrainingRunHistoryResponse,
)
from app.services.benchmark import benchmark_service
from app.services.evaluation import evaluation_service
from app.services.training import training_service


class AccuracyAuditService:
    def build_report(self, workspace_root: str) -> AccuracyAuditResponse:
        workspace_path = Path(workspace_root)
        # Accuracy audit needs a fresh readiness snapshot so installed dependencies
        # are reflected immediately in blocker reporting.
        readiness = evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=str(workspace_path),
                include_arms=list(evaluation_service.ARM_ROOTS.keys()),
                refresh_report=True,
            )
        )
        evidence = benchmark_service.load_report(workspace_path)
        training_runs = self._safe_training_runs(workspace_path)
        segmentation_bootstrap = self._load_segmentation_bootstrap(workspace_path)
        latest_by_arm = {}
        for run in training_runs.runs:
            if run.arm not in latest_by_arm:
                latest_by_arm[run.arm] = run

        raw_arm = next((arm for arm in readiness.arms if arm.arm_name == "raw160"), None)
        current_label_mode = self._detect_label_mode(Path(raw_arm.dataset_root) if raw_arm else workspace_path)
        detection_ready = current_label_mode in {"yolo_bbox", "yolo_segmentation"}
        segmentation_ready = current_label_mode in {"yolo_segmentation", "mask"}
        segmentation_bootstrap_dataset_root = workspace_path / "datasets" / "segmentation_bootstrap" / "coarse_masks"
        segmentation_bootstrap_ready = segmentation_bootstrap is not None or (
            (segmentation_bootstrap_dataset_root / "labels").exists()
            and any((segmentation_bootstrap_dataset_root / "labels").rglob("*.txt"))
            and (segmentation_bootstrap_dataset_root / "masks").exists()
        )
        segmentation_bootstrap_mode = segmentation_bootstrap.get("bootstrap_mode") if segmentation_bootstrap else None
        segmentation_bootstrap_refined_items = int(segmentation_bootstrap.get("refined_items", 0)) if segmentation_bootstrap else 0
        segmentation_bootstrap_dataset_root_str = (
            str(segmentation_bootstrap_dataset_root)
            if segmentation_bootstrap_ready
            else None
        )

        arms: list[AccuracyAuditArm] = []
        evidence_map = {arm.arm_name: arm for arm in evidence.arms}
        experiment_order = self._experiment_order(readiness.arms, evidence.recommended_arm)
        for priority, arm_name in enumerate(experiment_order, start=1):
            readiness_arm = next((arm for arm in readiness.arms if arm.arm_name == arm_name), None)
            if readiness_arm is None:
                continue
            evidence_arm = evidence_map.get("retinex80" if arm_name == "retinex" else arm_name)
            latest_run = latest_by_arm.get(arm_name)
            rationale = self._arm_rationale(arm_name, readiness_arm.notes, evidence_arm.evidence_score if evidence_arm else None, latest_run.metrics if latest_run else {})
            arms.append(
                AccuracyAuditArm(
                    arm_name=arm_name,
                    label_mode=self._detect_label_mode(Path(readiness_arm.dataset_root)),
                    split_image_counts=readiness_arm.split_image_counts,
                    evidence_score=evidence_arm.evidence_score if evidence_arm else None,
                    latest_metric_summary=self._latest_metric_summary(latest_run.metrics if latest_run else {}),
                    experiment_priority=priority,
                    status=latest_run.status if latest_run else ("ready" if readiness_arm.ready else "pending"),
                    rationale=rationale,
                )
            )

        blockers = []
        blockers.extend(self._requirement_blockers(readiness))
        if not segmentation_ready:
            blockers.append("Segmentation masks or polygon labels are not available yet; current workspace only supports direct detection accuracy checks.")
        if segmentation_bootstrap_ready and not segmentation_ready:
            blockers.append("A coarse segmentation bootstrap dataset exists, but it still requires review before true segmentation metrics can be claimed.")
        if segmentation_bootstrap_refined_items > 0 and not segmentation_ready:
            blockers.append("SAM-refined masks are available as bootstrap seeds, but reviewed gold masks are still required before claiming segmentation accuracy.")
        if any(arm.split_image_counts.get("val", 0) == 0 for arm in readiness.arms if arm.ready):
            blockers.append("Validation splits are empty in one or more arms, so current accuracy claims are not publication-grade yet.")

        dataset_train_images = raw_arm.split_image_counts.get("train", 0) if raw_arm else 0
        dataset_val_images = raw_arm.split_image_counts.get("val", 0) if raw_arm else 0
        dataset_test_images = raw_arm.split_image_counts.get("test", 0) if raw_arm else 0
        baseline_arm = "raw160"
        baseline_metric_summary = self._latest_metric_summary(latest_by_arm.get("raw160").metrics if latest_by_arm.get("raw160") else {})
        accuracy_readiness_score = self._accuracy_score(
            detection_ready=detection_ready,
            segmentation_ready=segmentation_ready,
            segmentation_bootstrap_ready=segmentation_bootstrap_ready,
            train_count=dataset_train_images,
            val_count=dataset_val_images,
            test_count=dataset_test_images,
            has_live_metric=baseline_metric_summary is not None,
            blocker_count=len(blockers),
        )

        report_root = workspace_path / "evaluations" / "accuracy_audit"
        report_root.mkdir(parents=True, exist_ok=True)
        response = AccuracyAuditResponse(
            workspace_root=str(workspace_path),
            current_label_mode=current_label_mode,
            detection_ready=detection_ready,
            segmentation_ready=segmentation_ready,
            segmentation_bootstrap_ready=segmentation_bootstrap_ready,
            segmentation_bootstrap_dataset_root=segmentation_bootstrap_dataset_root_str,
            segmentation_bootstrap_mode=segmentation_bootstrap_mode,
            segmentation_bootstrap_refined_items=segmentation_bootstrap_refined_items,
            accuracy_readiness_score=accuracy_readiness_score,
            dataset_train_images=dataset_train_images,
            dataset_val_images=dataset_val_images,
            dataset_test_images=dataset_test_images,
            baseline_arm=baseline_arm,
            baseline_metric_summary=baseline_metric_summary,
            primary_eval_metrics=self._primary_metrics(segmentation_ready),
            first_experiments=self._first_experiments(
                segmentation_ready,
                segmentation_bootstrap_ready,
                segmentation_bootstrap_mode,
                segmentation_bootstrap_refined_items,
            ),
            arms=arms,
            pixel_methods=self._pixel_method_plan(
                segmentation_ready,
                segmentation_bootstrap_ready,
                segmentation_bootstrap_mode,
                segmentation_bootstrap_refined_items,
            ),
            blockers=blockers,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> AccuracyAuditResponse:
        report_path = Path(workspace_root) / "evaluations" / "accuracy_audit" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing accuracy audit report: {report_path}")
        return AccuracyAuditResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _detect_label_mode(self, dataset_root: Path) -> str:
        label_root = dataset_root / "labels"
        if (dataset_root / "masks").exists():
            return "mask"
        if not label_root.exists():
            return "missing"
        sample = next(label_root.rglob("*.txt"), None)
        if sample is None:
            return "missing"
        for line in sample.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 5:
                return "yolo_bbox"
            if len(parts) > 5:
                return "yolo_segmentation"
        return "unknown"

    def _requirement_blockers(self, readiness) -> list[str]:
        blockers = []
        for requirement in readiness.requirements:
            if requirement.category == "program" and requirement.required:
                blockers.append(f"Required dependency missing: {requirement.name} ({requirement.reason})")
        return blockers

    def _primary_metrics(self, segmentation_ready: bool) -> list[str]:
        metrics = ["mAP50", "mAP50-95", "defect recall", "small-defect recall", "precision"]
        if segmentation_ready:
            metrics.extend(["mIoU", "Dice", "boundary F1"])
        else:
            metrics.append("segmentation mIoU (blocked until masks exist)")
        return metrics

    def _first_experiments(
        self,
        segmentation_ready: bool,
        segmentation_bootstrap_ready: bool = False,
        segmentation_bootstrap_mode: str | None = None,
        segmentation_bootstrap_refined_items: int = 0,
    ) -> list[str]:
        experiments = [
            "Train raw160 as the fixed baseline and record mAP50, recall, and precision.",
            "Train retinex next and compare gains against raw160 on the same frozen split.",
            "Train MergeMertens as the exposure-fusion baseline and compare against retinex.",
            "Train DAF last and test whether defect-preserving fusion beats plain Mertens on small-defect recall.",
            "Use the visual pipeline report to capture before/after figures from the same failure cases.",
        ]
        if segmentation_ready:
            experiments.append("Repeat the same arm order with segmentation metrics mIoU, Dice, and boundary F1.")
        else:
            experiments.append("Delay segmentation claims until mask or polygon labels are available.")
        if segmentation_bootstrap_ready:
            if segmentation_bootstrap_mode == "sam_refined_mask" and segmentation_bootstrap_refined_items > 0:
                experiments.append("Use the SAM-refined segmentation bootstrap as a review-required seed set, then promote a small gold subset before measuring mIoU.")
            else:
                experiments.append("Use the coarse segmentation bootstrap as a review-required seed set, then promote a small gold subset before measuring mIoU.")
        return experiments

    def _pixel_method_plan(
        self,
        segmentation_ready: bool,
        segmentation_bootstrap_ready: bool = False,
        segmentation_bootstrap_mode: str | None = None,
        segmentation_bootstrap_refined_items: int = 0,
    ) -> list[PixelMethodPlan]:
        seg_note = "Track mIoU and Dice after masks are available." if segmentation_ready else "Segmentation metrics remain blocked until mask labels exist."
        return [
            PixelMethodPlan(
                method_name="Retinex MSRCR",
                implemented=True,
                readiness="ready-now",
                accuracy_hypothesis="Improves low-light defect visibility and raises recall before full exposure fusion is introduced.",
                metrics_to_track=["mAP50", "defect recall", "small-defect recall"],
                next_step="Run as the first enhancement baseline on the frozen split.",
            ),
            PixelMethodPlan(
                method_name="MergeMertens Exposure Fusion",
                implemented=True,
                readiness="ready-now",
                accuracy_hypothesis="Combines low and high lux inputs into a more balanced representation that should improve robustness across exposure shifts.",
                metrics_to_track=["mAP50", "precision", "cross-lux generalization"],
                next_step="Use as the classical exposure-fusion baseline before claiming defect-aware gains.",
            ),
            PixelMethodPlan(
                method_name="Defect-Aware Fusion",
                implemented=True,
                readiness="ready-now",
                accuracy_hypothesis="Preserves high-frequency defect evidence better than plain exposure fusion and should improve small-defect recall.",
                metrics_to_track=["small-defect recall", "mAP50", "false positive change"],
                next_step="Run after Mertens and report the gain as the main paper contribution.",
            ),
            PixelMethodPlan(
                method_name="Local Relighting And Region Exposure Control",
                implemented=False,
                readiness="next-phase",
                accuracy_hypothesis="Will allow target-region brightness normalization without overexposing the whole frame, which is useful for very localized defects.",
                metrics_to_track=["defect recall", "background suppression", "boundary quality"],
                next_step="Implement region-specific relighting after the current detection audit is stabilized.",
            ),
            PixelMethodPlan(
                method_name="Segmentation Auto-Label And Mask Audit",
                implemented=segmentation_bootstrap_ready,
                readiness=(
                    "sam-refined-review-required"
                    if segmentation_bootstrap_mode == "sam_refined_mask" and segmentation_bootstrap_refined_items > 0 and not segmentation_ready
                    else ("review-required-bootstrap" if segmentation_bootstrap_ready and not segmentation_ready else ("blocked-by-labels" if not segmentation_ready else "ready-next"))
                ),
                accuracy_hypothesis="Uses bbox-derived coarse masks first, then upgrades to true segmentation evaluation only after reviewed masks or polygon labels exist.",
                metrics_to_track=["mIoU", "Dice", "boundary F1", seg_note],
                next_step=(
                    "Review the SAM-refined masks and promote a gold subset before claiming segmentation accuracy."
                    if segmentation_bootstrap_mode == "sam_refined_mask" and segmentation_bootstrap_refined_items > 0
                    else "Review the coarse bootstrap masks or add a SAM-style mask generator before claiming segmentation accuracy."
                ),
            ),
        ]

    def _arm_rationale(self, arm_name: str, notes: list[str], evidence_score: float | None, metrics: dict) -> list[str]:
        rationale = []
        if arm_name == "raw160":
            rationale.append("Use as the fixed baseline so every pixel method is compared against the same reference.")
        if evidence_score is not None:
            rationale.append(f"Current evidence score: {evidence_score}.")
        metric_summary = self._latest_metric_summary(metrics)
        if metric_summary:
            rationale.append(f"Latest recorded downstream metric: {metric_summary}.")
        rationale.extend(notes[:2])
        if not rationale:
            rationale.append("No additional rationale recorded yet.")
        return rationale

    def _experiment_order(self, arms, evidence_recommended_arm: str | None) -> list[str]:
        order = ["raw160"]
        if evidence_recommended_arm == "retinex80":
            order.append("retinex")
        for candidate in ["mertens", "daf", "retinex"]:
            if candidate not in order and any(arm.arm_name == candidate for arm in arms):
                order.append(candidate)
        return order

    def _latest_metric_summary(self, metrics: dict[str, float | int | str]) -> str | None:
        if not metrics:
            return None
        for key in ["mIoU", "Dice", "boundary F1", "boundary_f1", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/recall(B)", "metrics/precision(B)"]:
            if key in metrics:
                return f"{key}={metrics[key]}"
        first_key = next(iter(metrics.keys()), None)
        return f"{first_key}={metrics[first_key]}" if first_key else None

    def _accuracy_score(
        self,
        *,
        detection_ready: bool,
        segmentation_ready: bool,
        segmentation_bootstrap_ready: bool,
        train_count: int,
        val_count: int,
        test_count: int,
        has_live_metric: bool,
        blocker_count: int,
    ) -> int:
        score = 0
        if detection_ready:
            score += 35
        if segmentation_ready:
            score += 20
        elif segmentation_bootstrap_ready:
            score += 8
        score += min(20, train_count // 3)
        score += 10 if val_count > 0 else 0
        score += 8 if test_count > 0 else 0
        if has_live_metric:
            score += 12
        score -= blocker_count * 6
        return max(0, min(100, score))

    def _safe_training_runs(self, workspace_path: Path) -> TrainingRunHistoryResponse:
        try:
            return training_service.list_training_runs(str(workspace_path))
        except FileNotFoundError:
            return TrainingRunHistoryResponse(workspace_root=str(workspace_path), total_runs=0, runs=[])

    def _load_segmentation_bootstrap(self, workspace_path: Path) -> dict | None:
        report_path = workspace_path / "evaluations" / "segmentation_bootstrap" / "report.json"
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_report(self, report: AccuracyAuditResponse) -> None:
        report_json_path = Path(report.report_json_path)
        atomic_write_text(report_json_path, json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        lines = [
            "# Accuracy Audit",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Label mode: **{report.current_label_mode}**",
            f"- Detection ready: **{report.detection_ready}**",
            f"- Segmentation ready: **{report.segmentation_ready}**",
            f"- Segmentation bootstrap ready: **{report.segmentation_bootstrap_ready}**",
            f"- Segmentation bootstrap dataset: `{report.segmentation_bootstrap_dataset_root or 'n/a'}`",
            f"- Segmentation bootstrap mode: **{report.segmentation_bootstrap_mode or 'n/a'}**",
            f"- Segmentation bootstrap refined items: **{report.segmentation_bootstrap_refined_items}**",
            f"- Accuracy readiness score: **{report.accuracy_readiness_score}/100**",
            f"- Baseline arm: **{report.baseline_arm or 'n/a'}**",
            f"- Baseline metric: **{report.baseline_metric_summary or 'n/a'}**",
            "",
            "## Primary Metrics",
        ]
        for item in report.primary_eval_metrics:
            lines.append(f"- {item}")
        lines.extend(["", "## First Experiments"])
        for item in report.first_experiments:
            lines.append(f"- {item}")
        lines.extend(["", "## Arm Audit"])
        for arm in report.arms:
            lines.append(
                f"- `{arm.arm_name}`: priority={arm.experiment_priority}, status={arm.status}, "
                f"label_mode={arm.label_mode}, evidence={arm.evidence_score}, metric={arm.latest_metric_summary or 'n/a'}"
            )
            for item in arm.rationale:
                lines.append(f"  - {item}")
        lines.extend(["", "## Pixel Methods"])
        for method in report.pixel_methods:
            lines.append(
                f"- `{method.method_name}`: implemented={method.implemented}, readiness={method.readiness}, "
                f"hypothesis={method.accuracy_hypothesis}"
            )
            lines.append(f"  - next: {method.next_step}")
            lines.append(f"  - metrics: {', '.join(method.metrics_to_track)}")
        lines.extend(["", "## Blockers"])
        for item in report.blockers:
            lines.append(f"- {item}")
        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n")


accuracy_audit_service = AccuracyAuditService()
