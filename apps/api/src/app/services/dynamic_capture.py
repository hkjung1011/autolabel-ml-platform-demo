from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import DynamicCapturePlanItem, DynamicCaptureRequest, DynamicCaptureResponse, PairGroup


class DynamicCaptureService:
    def build_report(self, request: DynamicCaptureRequest) -> DynamicCaptureResponse:
        workspace_path = Path(request.workspace_root)
        groups = self._load_groups(workspace_path)

        plans = [self._plan_group(group, request.target_mid_lux, request.preferred_source_luxes) for group in groups]
        bracket_ready_groups = sum(1 for plan in plans if plan.capture_mode != "single_capture")
        high_dynamic_range_groups = sum(1 for plan in plans if plan.risk_level in {"high", "critical"})
        forensic_followup_groups = sum(1 for plan in plans if plan.next_pixel_branch == "forensic_wdr")

        recommended_global_mode = "single_capture"
        if forensic_followup_groups >= max(1, len(plans) // 3):
            recommended_global_mode = "forensic_wdr_bracket_3"
        elif bracket_ready_groups >= max(1, len(plans) // 3):
            recommended_global_mode = "adaptive_bracket_capture"

        report_root = workspace_path / "evaluations" / "dynamic_capture"
        report_root.mkdir(parents=True, exist_ok=True)
        response = DynamicCaptureResponse(
            workspace_root=str(workspace_path),
            target_mid_lux=request.target_mid_lux,
            recommended_global_mode=recommended_global_mode,
            total_groups=len(plans),
            bracket_ready_groups=bracket_ready_groups,
            high_dynamic_range_groups=high_dynamic_range_groups,
            forensic_followup_groups=forensic_followup_groups,
            plans=plans[: max(1, request.max_preview_groups)],
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> DynamicCaptureResponse:
        path = Path(workspace_root) / "evaluations" / "dynamic_capture" / "report.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing dynamic capture report: {path}")
        return DynamicCaptureResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_groups(self, workspace_path: Path) -> list[PairGroup]:
        for candidate in [
            workspace_path / "manifests" / "labeled_pair_manifest.json",
            workspace_path / "manifests" / "pair_manifest.json",
        ]:
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                return [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        raise FileNotFoundError(f"Missing pair manifest under {workspace_path / 'manifests'}")

    def _plan_group(self, group: PairGroup, target_mid_lux: int, preferred_source_luxes: list[str]) -> DynamicCapturePlanItem:
        available = sorted((int(lux) for lux in group.exposures.keys()), key=int)
        available_str = [str(value) for value in available]
        min_lux = available[0] if available else target_mid_lux
        max_lux = available[-1] if available else target_mid_lux

        capture_mode = "single_capture"
        exposure_plan = [f"{target_mid_lux} lux target"]
        next_pixel_branch = "retinex"
        risk_level = "low"
        notes: list[str] = []

        has_shadow = min_lux <= 40
        has_highlight = max_lux >= 160
        multi_exposure = len(available) >= 3

        if has_shadow and has_highlight and multi_exposure:
            capture_mode = "forensic_wdr_bracket_3"
            exposure_plan = [f"{lux} lux" for lux in available if str(lux) in preferred_source_luxes] or [f"{lux} lux" for lux in available[:3]]
            next_pixel_branch = "forensic_wdr"
            risk_level = "critical"
            notes.append("High dynamic range scene: retain both highlight and shadow evidence.")
            notes.append("Recommended follow-up: Forensic WDR before detector or auto-labeling.")
        elif has_shadow and len(available) >= 2:
            capture_mode = "shadow_recovery_bracket"
            exposure_plan = [f"{lux} lux" for lux in available[:2]]
            next_pixel_branch = "retinex"
            risk_level = "high"
            notes.append("Low-light risk detected: capture a darker and a mid exposure for shadow recovery.")
        elif has_highlight and len(available) >= 2:
            capture_mode = "highlight_protect_bracket"
            exposure_plan = [f"{lux} lux" for lux in available[-2:]]
            next_pixel_branch = "target_lux"
            risk_level = "medium"
            notes.append("Highlight clipping risk detected: capture a mid and bright exposure to preserve reflective surfaces.")
        elif len(available) >= 2:
            capture_mode = "adaptive_bracket_capture"
            exposure_plan = [f"{lux} lux" for lux in available[:2]]
            next_pixel_branch = "mertens"
            risk_level = "medium"
            notes.append("Use paired exposures for safer fusion and comparison.")
        else:
            notes.append("Single exposure is acceptable, but keep target mid-tone near the defect region.")

        if group.label_reusable:
            notes.append("Anchor labels already exist, so capture variants can be benchmarked immediately.")
        if not group.label_reusable:
            notes.append("No reusable anchor label yet; route through auto-label and review after capture.")

        return DynamicCapturePlanItem(
            group_id=group.key,
            split=group.frozen_split or group.split,
            available_luxes=available_str,
            capture_mode=capture_mode,
            exposure_plan=exposure_plan,
            next_pixel_branch=next_pixel_branch,
            risk_level=risk_level,
            notes=notes,
            anchor_image_path=group.anchor_image_path,
        )

    def _write_report(self, report: DynamicCaptureResponse) -> None:
        atomic_write_text(
            report.report_json_path,
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Dynamic Capture Planner",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Target mid lux: **{report.target_mid_lux}**",
            f"- Recommended global mode: **{report.recommended_global_mode}**",
            f"- Total groups: **{report.total_groups}**",
            f"- Bracket-ready groups: **{report.bracket_ready_groups}**",
            f"- High dynamic range groups: **{report.high_dynamic_range_groups}**",
            f"- Forensic follow-up groups: **{report.forensic_followup_groups}**",
            "",
            "## Preview Plans",
        ]
        for plan in report.plans:
            lines.append(
                f"- `{plan.group_id}` [{plan.split}] mode={plan.capture_mode}, next={plan.next_pixel_branch}, risk={plan.risk_level}, lux={', '.join(plan.available_luxes)}"
            )
            for note in plan.notes:
                lines.append(f"  - {note}")
        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")


dynamic_capture_service = DynamicCaptureService()
