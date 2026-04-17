from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import OperatorGuideResponse, OperatorGuideStep
from app.services.desktop_runtime import desktop_runtime_service
from app.services.program_status import program_status_service


class OperatorGuideService:
    def build_report(self, workspace_root: str) -> OperatorGuideResponse:
        workspace_path = Path(workspace_root)
        program_status = program_status_service.build_report(str(workspace_path))
        runtime = desktop_runtime_service.build_report(str(workspace_path))

        steps = [
            OperatorGuideStep(
                step_name="Workspace Bootstrap",
                status="implemented" if program_status.overall_progress_percent >= 60 else "partial",
                progress_percent=min(100, max(35, program_status.overall_progress_percent)),
                summary="Workspace manifests, candidate splits, and local experiment folders should exist before any operator workflow starts.",
                recommended_action="Build or load the workspace and confirm the correct staged dataset root.",
                ui_anchor="#workspaceSection",
                api_hint="POST /api/research/v1/bootstrap",
            ),
            OperatorGuideStep(
                step_name="Desktop Runtime Diagnostics",
                status="implemented" if runtime.readiness_score >= 80 else ("partial" if runtime.readiness_score >= 50 else "warn"),
                progress_percent=runtime.readiness_score,
                summary="Executable and dependency health determine whether the app can move from demo mode into operator mode.",
                recommended_action=(runtime.next_actions[0] if runtime.next_actions else "Run desktop diagnostics and clear the remaining blockers."),
                ui_anchor="#runtimeSection",
                api_hint="POST /api/research/v1/desktop/runtime-check",
            ),
            OperatorGuideStep(
                step_name="Commercial Intake",
                status="implemented" if program_status.protected_source_count > 0 else "partial",
                progress_percent=100 if program_status.protected_source_count > 0 else 40,
                summary="External HDD datasets must stay read-only and be staged by copy into local candidate workspaces.",
                recommended_action="Build the protected source catalog and use Stage Copy instead of operating on the original dataset root.",
                ui_anchor="#commercialSection",
                api_hint="POST /api/research/v1/commercial/source-catalog",
            ),
            OperatorGuideStep(
                step_name="Detection Baseline",
                status="implemented" if program_status.detection_progress_percent >= 85 else "partial",
                progress_percent=program_status.detection_progress_percent,
                summary="Detection metrics and evidence ranking are the first trustworthy validation loop before segmentation claims.",
                recommended_action="Regenerate readiness, evidence, and training plans on the current staged workspace.",
                ui_anchor="#pipelineSection",
                api_hint="POST /api/research/v1/workspace/run-full",
            ),
            OperatorGuideStep(
                step_name="Auto-Label And Review",
                status="implemented" if program_status.autolabel_progress_percent >= 75 else "partial",
                progress_percent=program_status.autolabel_progress_percent,
                summary="Proposal generation is not enough; the operator loop needs review ownership and approved export for retraining.",
                recommended_action="Build auto-label proposals, review the highest-priority items, and export the approved set.",
                ui_anchor="#reviewSection",
                api_hint="POST /api/research/v1/autolabel/bootstrap",
            ),
            OperatorGuideStep(
                step_name="Segmentation Expansion",
                status="implemented" if program_status.segmentation_progress_percent >= 70 else "blocked",
                progress_percent=program_status.segmentation_progress_percent,
                summary="Mask bootstrap is still the main product gap between research usability and a stronger commercial segmentation tool.",
                recommended_action="Add a mask gold set or SAM-style bootstrap before promising segmentation accuracy.",
                ui_anchor="#statusSection",
                api_hint="POST /api/research/v1/reporting/accuracy-audit",
            ),
            OperatorGuideStep(
                step_name="EXE Packaging And Delivery",
                status="implemented" if runtime.exe_exists and runtime.package_build_ready else "partial",
                progress_percent=100 if runtime.exe_exists and runtime.package_build_ready else 60 if runtime.exe_exists else 30,
                summary="A pilot user should be able to launch the tool as a branded EXE without touching the dev environment.",
                recommended_action="Review package plan, download the current EXE, and smoke-test it on the target operator PC.",
                ui_anchor="#packageSection",
                api_hint="POST /api/research/v1/desktop/package-plan",
            ),
            OperatorGuideStep(
                step_name="Reporting And Export",
                status="implemented" if program_status.production_score >= 50 else "partial",
                progress_percent=min(100, max(program_status.production_score, program_status.field_score)),
                summary="Stakeholders need scorecards, CSV exports, and clear arm comparisons to trust the results.",
                recommended_action="Build scorecard, CSV export, and paper pack after every major pipeline run.",
                ui_anchor="#reportingSection",
                api_hint="POST /api/research/v1/reporting/scorecard",
            ),
        ]

        blockers = list(runtime.blockers) + list(program_status.blockers[:3])
        next_actions = self._dedupe(
            list(runtime.next_actions)
            + list(program_status.next_actions)
        )[:6]

        operator_readiness_score = round(
            (program_status.execution_readiness_percent * 0.35)
            + (runtime.readiness_score * 0.35)
            + (program_status.field_score * 0.20)
            + (program_status.production_score * 0.10)
        )

        current_focus = "Segmentation bootstrap and real trainer integration" if program_status.segmentation_progress_percent < 50 else "Pilot hardening and operator rollout"
        primary_action = next_actions[0] if next_actions else "Refresh operator view and review the top blockers."

        report_root = workspace_path / "evaluations" / "operator_guide"
        report_root.mkdir(parents=True, exist_ok=True)
        response = OperatorGuideResponse(
            workspace_root=str(workspace_path),
            operator_readiness_score=operator_readiness_score,
            current_focus=current_focus,
            primary_action=primary_action,
            blockers=self._dedupe(blockers)[:6],
            next_actions=next_actions,
            steps=steps,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> OperatorGuideResponse:
        report_path = Path(workspace_root) / "evaluations" / "operator_guide" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing operator guide report: {report_path}")
        return OperatorGuideResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    def _write_report(self, response: OperatorGuideResponse) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Operator Guide",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Operator readiness: **{response.operator_readiness_score}/100**",
            f"- Current focus: **{response.current_focus}**",
            f"- Primary action: **{response.primary_action}**",
            "",
            "## Steps",
        ]
        for step in response.steps:
            lines.append(f"- `{step.step_name}` status={step.status}, progress={step.progress_percent}%")
            lines.append(f"  - summary: {step.summary}")
            lines.append(f"  - action: {step.recommended_action}")
            lines.append(f"  - ui: {step.ui_anchor}")
            if step.api_hint:
                lines.append(f"  - api: {step.api_hint}")
        lines.extend(["", "## Blockers"])
        for item in response.blockers:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in response.next_actions:
            lines.append(f"- {item}")
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")


operator_guide_service = OperatorGuideService()
