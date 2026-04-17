from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    EvaluationArmSummary,
    EvaluationReadinessReport,
    EvaluationRequirement,
    EvaluationRunRequest,
)


class EvaluationService:
    ARM_ROOTS = {
        "raw160": ("datasets", "yolo_baseline"),
        "retinex": ("datasets", "registered_variants", "retinex_msrcr"),
        "mertens": ("datasets", "yolo_fusion", "mertens_baseline"),
        "daf": ("datasets", "yolo_fusion", "defect_aware_fusion"),
        "mixed_raw_forensic_wdr": ("datasets", "yolo_mixed", "raw160_forensic_wdr"),
    }

    def build_readiness_report(self, request: EvaluationRunRequest) -> EvaluationReadinessReport:
        workspace_root = Path(request.workspace_root)
        if not workspace_root.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {workspace_root}")

        evaluations_root = workspace_root / "evaluations" / "readiness"
        evaluations_root.mkdir(parents=True, exist_ok=True)

        arms = [self._summarize_arm(workspace_root, arm_name) for arm_name in request.include_arms if arm_name in self.ARM_ROOTS]
        phase_status = self._build_phase_status(workspace_root, arms)
        completion_percent = self._compute_completion_percent(phase_status)
        requirements = self._build_requirements()
        execution_readiness_percent = self._compute_execution_readiness_percent(completion_percent, arms, requirements)

        report = EvaluationReadinessReport(
            workspace_root=str(workspace_root),
            completion_percent=completion_percent,
            execution_readiness_percent=execution_readiness_percent,
            phase_status=phase_status,
            arms=arms,
            requirements=requirements,
            report_json_path=str(evaluations_root / "report.json"),
            report_markdown_path=str(evaluations_root / "report.md"),
        )

        if request.refresh_report:
            self._write_report_files(report, evaluations_root)

        return report

    def load_readiness_report(self, workspace_root: str | Path) -> EvaluationReadinessReport:
        workspace_root = Path(workspace_root)
        report_path = workspace_root / "evaluations" / "readiness" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing readiness report: {report_path}")
        return EvaluationReadinessReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _summarize_arm(self, workspace_root: Path, arm_name: str) -> EvaluationArmSummary:
        dataset_root = workspace_root.joinpath(*self.ARM_ROOTS[arm_name])
        image_root = dataset_root / "images"
        label_root = dataset_root / "labels"

        split_image_counts = self._count_split_files(image_root, suffixes={".png", ".jpg", ".jpeg", ".bmp"})
        split_label_counts = self._count_split_files(label_root, suffixes={".txt"})
        class_ids = self._collect_class_ids(label_root)
        notes: list[str] = []
        ready = any(
            split_image_counts.get(split, 0) > 0 and split_label_counts.get(split, 0) > 0
            for split in ("train", "val", "test")
        )
        if not dataset_root.exists():
            notes.append("Dataset root is missing.")
        if ready and not split_image_counts.get("train"):
            notes.append("Train split is empty; structure exists but training is not ready yet.")
        if ready and not split_image_counts.get("val"):
            notes.append("Validation split is empty or missing.")
        if ready and not split_image_counts.get("test"):
            notes.append("Test split is empty or missing.")
        if not class_ids and label_root.exists():
            notes.append("No class ids could be inferred from labels.")

        dataset_yaml_path = None
        if ready:
            dataset_yaml_path = str(self._write_dataset_yaml(dataset_root, arm_name, class_ids))

        return EvaluationArmSummary(
            arm_name=arm_name,
            dataset_root=str(dataset_root),
            split_image_counts=split_image_counts,
            split_label_counts=split_label_counts,
            class_ids=class_ids,
            dataset_yaml_path=dataset_yaml_path,
            ready=ready,
            notes=notes,
        )

    def _count_split_files(self, root: Path, suffixes: set[str]) -> dict[str, int]:
        counts = {"train": 0, "val": 0, "test": 0}
        if not root.exists():
            return counts
        for split in counts:
            split_dir = root / split
            if split_dir.exists():
                counts[split] = sum(1 for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
        return counts

    def _collect_class_ids(self, label_root: Path) -> list[int]:
        class_ids: set[int] = set()
        if not label_root.exists():
            return []
        for label_path in label_root.rglob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_ids.add(int(float(parts[0])))
                except ValueError:
                    continue
        return sorted(class_ids)

    def _write_dataset_yaml(self, dataset_root: Path, arm_name: str, class_ids: list[int]) -> Path:
        yaml_root = dataset_root / "meta"
        yaml_root.mkdir(parents=True, exist_ok=True)
        yaml_path = yaml_root / f"{arm_name}.yaml"
        names = [f"class_{class_id}" for class_id in class_ids] or ["class_0"]
        lines = [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            "names:",
        ]
        lines.extend([f"  {index}: {name}" for index, name in enumerate(names)])
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def _build_phase_status(self, workspace_root: Path, arms: list[EvaluationArmSummary]) -> dict[str, str]:
        status = {
            "phase0_split_freeze": "done" if (workspace_root / "manifests" / "split_manifest.json").exists() else "missing",
            "phase1_bootstrap": "done" if (workspace_root / "manifests" / "summary.json").exists() else "missing",
            "phase2_retinex": "done" if (workspace_root / "variants" / "retinex_msrcr").exists() else "missing",
            "phase3_registration": "done" if (workspace_root / "registration_reports" / "retinex_msrcr_accepted_manifest.json").exists() else "missing",
            "phase4_mertens": "done" if any(arm.arm_name == "mertens" and arm.ready for arm in arms) else "missing",
            "phase5_daf": "done" if any(arm.arm_name == "daf" and arm.ready for arm in arms) else "missing",
            "phase6_evaluation": "done" if any(arm.ready for arm in arms) else "in_progress",
        }
        return status

    def _compute_completion_percent(self, phase_status: dict[str, str]) -> int:
        phase_weights = {
            "phase0_split_freeze": 12,
            "phase1_bootstrap": 14,
            "phase2_retinex": 14,
            "phase3_registration": 16,
            "phase4_mertens": 14,
            "phase5_daf": 14,
            "phase6_evaluation": 16,
        }
        total = 0
        for phase, weight in phase_weights.items():
            state = phase_status.get(phase, "missing")
            if state == "done":
                total += weight
            elif state == "in_progress":
                total += weight // 2
        return min(100, total)

    def _compute_execution_readiness_percent(
        self,
        completion_percent: int,
        arms: list[EvaluationArmSummary],
        requirements: list[EvaluationRequirement],
    ) -> int:
        missing_program_penalty = sum(
            8 for requirement in requirements if requirement.category == "program" and requirement.required
        )
        split_penalty = min(
            10,
            sum(
                2
                for arm in arms
                if arm.ready and any("Validation split is empty" in note or "Test split is empty" in note for note in arm.notes)
            ),
        )
        return max(0, completion_percent - missing_program_penalty - split_penalty)

    def _build_requirements(self) -> list[EvaluationRequirement]:
        ultralytics_installed = bool(importlib.util.find_spec("ultralytics"))
        opencv_installed = bool(importlib.util.find_spec("cv2"))
        return [
            EvaluationRequirement(
                category="program",
                name="ultralytics",
                required=not ultralytics_installed,
                reason="Required for Phase 6 YOLO training and cross-lux evaluation runs.",
            ),
            EvaluationRequirement(
                category="program",
                name="opencv-python",
                required=not opencv_installed,
                reason="Recommended for stronger ECC/ORB registration than the current lightweight aligner.",
            ),
            EvaluationRequirement(
                category="data",
                name="aligned_gold_pairs",
                required=False,
                reason="Helpful for tuning registration thresholds and validating defect-preservation metrics.",
            ),
        ]

    def _write_report_files(self, report: EvaluationReadinessReport, evaluations_root: Path) -> None:
        report_json_path = Path(report.report_json_path)
        atomic_write_text(report_json_path, json.dumps(report.model_dump(), ensure_ascii=False, indent=2))

        lines = [
            "# Evaluation Readiness Report",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Completion: **{report.completion_percent}%**",
            f"- Execution readiness: **{report.execution_readiness_percent}%**",
            "",
            "## Phase Status",
        ]
        for phase, status in report.phase_status.items():
            lines.append(f"- `{phase}`: {status}")

        lines.extend(["", "## Arms"])
        for arm in report.arms:
            lines.append(f"- `{arm.arm_name}`: ready={arm.ready}, images={arm.split_image_counts}, labels={arm.split_label_counts}")
            if arm.dataset_yaml_path:
                lines.append(f"  - yaml: `{arm.dataset_yaml_path}`")
            for note in arm.notes:
                lines.append(f"  - note: {note}")

        lines.extend(["", "## Requirements"])
        for requirement in report.requirements:
            lines.append(
                f"- `{requirement.category}` / `{requirement.name}`: required={requirement.required} - {requirement.reason}"
            )

        report_md_path = Path(report.report_markdown_path)
        atomic_write_text(report_md_path, "\n".join(lines) + "\n")


evaluation_service = EvaluationService()
