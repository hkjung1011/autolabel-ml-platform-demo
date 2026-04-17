from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    EvidenceArmSummary,
    EvidenceBenchmarkReport,
    EvidenceRunRequest,
    PairGroup,
)
from app.plugins.registration.verifier import read_yolo_boxes, to_xyxy


class BenchmarkService:
    def build_evidence_report(self, request: EvidenceRunRequest) -> EvidenceBenchmarkReport:
        workspace_root = Path(request.workspace_root)
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing labeled manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        evaluations_root = workspace_root / "evaluations" / "evidence"
        evaluations_root.mkdir(parents=True, exist_ok=True)

        arm_records: dict[str, list[dict[str, float]]] = {arm: [] for arm in request.compare_arms}
        common_group_count = 0
        for group in groups:
            if not group.anchor_label_path:
                continue
            paths = self._resolve_arm_paths(workspace_root, group, request.source_lux)
            selected_paths = {arm: path for arm, path in paths.items() if arm in request.compare_arms and path and Path(path).exists()}
            if len(selected_paths) < 2 or "raw80" not in selected_paths:
                continue
            common_group_count += 1
            boxes = read_yolo_boxes(Path(group.anchor_label_path))
            for arm_name, image_path in selected_paths.items():
                arm_records[arm_name].append(self._measure_image(Path(image_path), boxes))

        arms = self._summarize_arms(arm_records)
        recommended_arm, peak_arm, key_takeaways = self._recommend_arm(arms)
        report = EvidenceBenchmarkReport(
            workspace_root=str(workspace_root),
            source_lux=request.source_lux,
            common_group_count=common_group_count,
            arms=arms,
            recommended_arm=recommended_arm,
            peak_arm=peak_arm,
            key_takeaways=key_takeaways,
            report_json_path=str(evaluations_root / "report.json"),
            report_markdown_path=str(evaluations_root / "report.md"),
        )
        if request.refresh_report:
            self._write_report(report)
        return report

    def load_report(self, workspace_root: str | Path) -> EvidenceBenchmarkReport:
        workspace_root = Path(workspace_root)
        report_path = workspace_root / "evaluations" / "evidence" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing evidence report: {report_path}")
        return EvidenceBenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _resolve_arm_paths(self, workspace_root: Path, group: PairGroup, source_lux: str) -> dict[str, str | None]:
        safe_group = group.key.replace("|", "__")
        return {
            "raw80": group.exposures.get(source_lux),
            "retinex80": str(workspace_root / "variants" / "retinex_msrcr" / safe_group / f"lux{source_lux}_restored.png"),
            "mertens": str(workspace_root / "variants" / "mertens_baseline" / safe_group / "fused.png"),
            "daf": str(workspace_root / "variants" / "defect_aware_fusion" / safe_group / "fused.png"),
            "raw160": group.anchor_image_path,
        }

    def _measure_image(self, image_path: Path, boxes: list[tuple[float, float, float, float]]) -> dict[str, float]:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        gray = self._to_gray(image)
        dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        gradient = self._gradient(gray)

        if not boxes:
            return {
                "dynamic_range": dynamic_range,
                "defect_visibility": float(gradient.mean()),
                "background_suppression": 0.0,
            }

        height, width = gray.shape
        defect_scores = []
        background_scores = []
        mask = np.zeros_like(gray, dtype=bool)
        for box in boxes:
            x0, y0, x1, y1 = to_xyxy(*box, image_width=width, image_height=height)
            xi0 = max(0, min(width, int(round(x0))))
            yi0 = max(0, min(height, int(round(y0))))
            xi1 = max(0, min(width, int(round(x1))))
            yi1 = max(0, min(height, int(round(y1))))
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            region = gradient[yi0:yi1, xi0:xi1]
            defect_scores.append(float(region.mean()))
            mask[yi0:yi1, xi0:xi1] = True

        background_region = gradient[~mask]
        if background_region.size:
            background_scores.append(float(background_region.mean()))

        defect_visibility = float(np.mean(defect_scores)) if defect_scores else float(gradient.mean())
        background_visibility = float(np.mean(background_scores)) if background_scores else 0.0
        return {
            "dynamic_range": dynamic_range,
            "defect_visibility": defect_visibility,
            "background_suppression": defect_visibility - background_visibility,
        }

    def _summarize_arms(self, arm_records: dict[str, list[dict[str, float]]]) -> list[EvidenceArmSummary]:
        baseline_visibility = None
        baseline_dynamic = None
        summaries: list[EvidenceArmSummary] = []
        max_groups = max((len(records) for records in arm_records.values()), default=0)

        if arm_records.get("raw80"):
            baseline_visibility = float(np.mean([item["defect_visibility"] for item in arm_records["raw80"]]))
            baseline_dynamic = float(np.mean([item["dynamic_range"] for item in arm_records["raw80"]]))

        for arm_name, records in arm_records.items():
            if not records:
                summaries.append(
                    EvidenceArmSummary(
                        arm_name=arm_name,
                        available_groups=0,
                        avg_dynamic_range=0.0,
                        avg_defect_visibility=0.0,
                        avg_background_suppression=0.0,
                        coverage_ratio=0.0,
                        evidence_score=0.0,
                        notes=["No comparable groups available."],
                    )
                )
                continue
            avg_dynamic = float(np.mean([item["dynamic_range"] for item in records]))
            avg_visibility = float(np.mean([item["defect_visibility"] for item in records]))
            avg_suppression = float(np.mean([item["background_suppression"] for item in records]))
            coverage_ratio = (len(records) / max_groups) if max_groups else 0.0
            evidence_score = (avg_dynamic + avg_visibility + avg_suppression) * coverage_ratio
            notes: list[str] = []
            if coverage_ratio < 0.8:
                notes.append("Coverage is smaller than the raw80 baseline; treat this branch as an early sample result.")
            if baseline_visibility is not None and avg_visibility > baseline_visibility:
                notes.append(f"Defect visibility proxy improved by {round(avg_visibility - baseline_visibility, 4)} vs raw80.")
            if baseline_dynamic is not None and avg_dynamic > baseline_dynamic:
                notes.append(f"Dynamic range proxy improved by {round(avg_dynamic - baseline_dynamic, 4)} vs raw80.")
            summaries.append(
                EvidenceArmSummary(
                    arm_name=arm_name,
                    available_groups=len(records),
                    avg_dynamic_range=round(avg_dynamic, 4),
                    avg_defect_visibility=round(avg_visibility, 4),
                    avg_background_suppression=round(avg_suppression, 4),
                    coverage_ratio=round(coverage_ratio, 4),
                    evidence_score=round(evidence_score, 4),
                    visibility_gain_vs_raw80=(
                        round(avg_visibility - baseline_visibility, 4) if baseline_visibility is not None else None
                    ),
                    dynamic_range_gain_vs_raw80=(
                        round(avg_dynamic - baseline_dynamic, 4) if baseline_dynamic is not None else None
                    ),
                    notes=notes,
                )
            )
        return summaries

    def _gradient(self, gray: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(gray)
        return np.sqrt((gx**2) + (gy**2))

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])

    def _write_report(self, report: EvidenceBenchmarkReport) -> None:
        report_json_path = Path(report.report_json_path)
        atomic_write_text(report_json_path, json.dumps(report.model_dump(), ensure_ascii=False, indent=2))

        lines = [
            "# Evidence Benchmark Report",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Source lux: `{report.source_lux}`",
            f"- Comparable groups: **{report.common_group_count}**",
            f"- Recommended arm: **{report.recommended_arm or 'n/a'}**",
            f"- Peak arm: **{report.peak_arm or 'n/a'}**",
            "",
            "## Takeaways",
        ]
        for takeaway in report.key_takeaways:
            lines.append(f"- {takeaway}")

        lines.extend([
            "",
            "## Arms",
        ])
        for arm in report.arms:
            lines.append(
                f"- `{arm.arm_name}`: groups={arm.available_groups}, visibility={arm.avg_defect_visibility}, "
                f"range={arm.avg_dynamic_range}, suppression={arm.avg_background_suppression}, "
                f"coverage={arm.coverage_ratio}, score={arm.evidence_score}, "
                f"visibility_gain_vs_raw80={arm.visibility_gain_vs_raw80}, dynamic_gain_vs_raw80={arm.dynamic_range_gain_vs_raw80}"
            )
            for note in arm.notes:
                lines.append(f"  - note: {note}")

        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n")

    def _recommend_arm(self, arms: list[EvidenceArmSummary]) -> tuple[str | None, str | None, list[str]]:
        eligible = [arm for arm in arms if arm.available_groups > 0]
        if not eligible:
            return None, None, ["No comparable groups were available for evidence ranking."]

        peak = max(
            eligible,
            key=lambda arm: (
                arm.avg_defect_visibility + arm.avg_background_suppression + arm.avg_dynamic_range,
                arm.available_groups,
            ),
        )
        stable_candidates = [arm for arm in eligible if (arm.coverage_ratio or 0.0) >= 0.8]
        best = max(
            stable_candidates or eligible,
            key=lambda arm: (
                arm.evidence_score or 0.0,
                arm.avg_defect_visibility,
                arm.available_groups,
            ),
        )
        takeaways = [
            f"`{best.arm_name}` currently has the strongest combined defect-visibility and dynamic-range proxy score.",
        ]
        if peak.arm_name != best.arm_name:
            takeaways.append(
                f"`{peak.arm_name}` has the strongest peak score, but its comparable subset is smaller than the recommended branch."
            )
        baseline = next((arm for arm in eligible if arm.arm_name == "raw80"), None)
        if baseline and best.arm_name != "raw80":
            takeaways.append(
                f"`{best.arm_name}` improves defect visibility by {best.visibility_gain_vs_raw80} over `raw80` in the current comparable subset."
            )
        if any(arm.available_groups < (baseline.available_groups if baseline else arm.available_groups) for arm in eligible):
            takeaways.append("Some fusion branches still have smaller comparable subsets than `raw80`; the ranking should be rechecked after full-batch generation.")
        return best.arm_name, peak.arm_name, takeaways


benchmark_service = BenchmarkService()
