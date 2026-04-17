from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    EvaluationRunRequest,
    FusionRunRequest,
    RegistrationVerifyRequest,
    RetinexRunRequest,
    TrainingRunRequest,
    WorkspacePipelineRunRequest,
    WorkspacePipelineRunResponse,
    WorkspacePipelineStage,
)
from app.services.benchmark import benchmark_service
from app.services.evaluation import evaluation_service
from app.services.fusion_runner import fusion_runner_service
from app.services.registration import registration_service
from app.services.retinex_runner import retinex_runner_service
from app.services.training import training_service
from app.domain.research_models import EvidenceRunRequest


class WorkspaceRunnerService:
    def run_full_pipeline(self, request: WorkspacePipelineRunRequest) -> WorkspacePipelineRunResponse:
        workspace_root = Path(request.workspace_root)
        stages: list[WorkspacePipelineStage] = []

        retinex_job = retinex_runner_service.run(
            RetinexRunRequest(
                workspace_root=request.workspace_root,
                source_luxes=request.retinex_luxes,
                method="msrcr",
                overwrite=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="retinex",
                status=retinex_job.status,
                summary=f"{len(retinex_job.outputs)} outputs, {len(retinex_job.errors)} errors",
                artifact_path=retinex_job.output_root,
                preview_paths=self._preview_paths_for_retinex(retinex_job),
            )
        )

        registration_job = registration_service.run(
            RegistrationVerifyRequest(
                workspace_root=request.workspace_root,
                variant_source="retinex_msrcr",
                source_luxes=request.retinex_luxes,
                materialize_accepted_dataset=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="registration",
                status=registration_job.status,
                summary=(
                    f"accepted={registration_job.accepted_count}, "
                    f"warning={registration_job.warning_count}, reject={registration_job.rejected_count}"
                ),
                artifact_path=registration_job.accepted_manifest_path,
                preview_paths=self._preview_paths_for_registration(registration_job),
            )
        )

        mertens_job = fusion_runner_service.run_mertens(
            FusionRunRequest(
                workspace_root=request.workspace_root,
                source_luxes=request.fusion_luxes,
                method="mertens",
                overwrite=True,
                use_labeled_only=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="mertens",
                status=mertens_job.status,
                summary=f"{len(mertens_job.outputs)} outputs, {len(mertens_job.errors)} errors",
                artifact_path=mertens_job.dataset_root,
                preview_paths=self._preview_paths_for_fusion(mertens_job),
            )
        )

        daf_job = fusion_runner_service.run_daf(
            FusionRunRequest(
                workspace_root=request.workspace_root,
                source_luxes=request.fusion_luxes,
                method="daf",
                overwrite=True,
                use_labeled_only=True,
                emit_debug_artifacts=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="daf",
                status=daf_job.status,
                summary=f"{len(daf_job.outputs)} outputs, {len(daf_job.errors)} errors",
                artifact_path=daf_job.dataset_root,
                preview_paths=self._preview_paths_for_fusion(daf_job),
            )
        )

        readiness = evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=request.workspace_root,
                include_arms=["raw160", "retinex", "mertens", "daf"],
                refresh_report=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="readiness",
                status="completed",
                summary=(
                    f"completion={readiness.completion_percent}%, "
                    f"execution={readiness.execution_readiness_percent}%"
                ),
                artifact_path=readiness.report_json_path,
            )
        )

        evidence = benchmark_service.build_evidence_report(
            EvidenceRunRequest(
                workspace_root=request.workspace_root,
                source_lux=request.evidence_source_lux,
                compare_arms=request.evidence_compare_arms,
                refresh_report=True,
            )
        )
        stages.append(
            WorkspacePipelineStage(
                stage_name="evidence",
                status="completed",
                summary=(
                    f"recommended={evidence.recommended_arm or 'n/a'}, "
                    f"peak={evidence.peak_arm or 'n/a'}, groups={evidence.common_group_count}"
                ),
                artifact_path=evidence.report_json_path,
            )
        )

        if request.include_training_plan:
            training = training_service.run_training(
                TrainingRunRequest(
                    workspace_root=request.workspace_root,
                    arm=request.training_arm,
                    dry_run=request.training_dry_run,
                )
            )
            stages.append(
                WorkspacePipelineStage(
                    stage_name="training_plan",
                    status=training.status,
                    summary=f"{training.arm} {training.status}",
                    artifact_path=training.artifact_paths.get("plan_json_path"),
                )
            )

        overall_status = "completed"
        if any(stage.status in {"failed", "blocked"} for stage in stages):
            overall_status = "partial"

        report_paths = self._write_pipeline_report(workspace_root=workspace_root, stages=stages, response_status=overall_status)

        return WorkspacePipelineRunResponse(
            workspace_root=request.workspace_root,
            status=overall_status,
            completion_percent=readiness.completion_percent,
            execution_readiness_percent=readiness.execution_readiness_percent,
            recommended_arm=evidence.recommended_arm,
            peak_arm=evidence.peak_arm,
            stages=stages,
            report_json_path=str(report_paths["json"]),
            report_markdown_path=str(report_paths["markdown"]),
        )

    def _preview_paths_for_retinex(self, job) -> list[str]:
        if not job.outputs:
            return []
        sample = job.outputs[0]
        return [path for path in [sample.anchor_path, sample.output_path] if path]

    def _preview_paths_for_registration(self, job) -> list[str]:
        if not job.reports:
            return []
        sample = job.reports[0]
        return [path for path in [sample.source_path, sample.variant_path, sample.anchor_path] if path]

    def _preview_paths_for_fusion(self, job) -> list[str]:
        if not job.outputs:
            return []
        sample = job.outputs[0]
        preview_paths = [sample.output_path]
        preview_paths.extend(value for value in sample.weight_map_paths.values())
        preview_paths.extend(value for value in sample.artifact_paths.values())
        return preview_paths[:4]

    def _write_pipeline_report(
        self,
        workspace_root: Path,
        stages: list[WorkspacePipelineStage],
        response_status: str,
    ) -> dict[str, Path]:
        report_root = workspace_root / "evaluations" / "pipeline"
        report_root.mkdir(parents=True, exist_ok=True)
        json_path = report_root / "report.json"
        markdown_path = report_root / "report.md"
        payload = {
            "workspace_root": str(workspace_root),
            "status": response_status,
            "stage_count": len(stages),
            "stages": [stage.model_dump() for stage in stages],
        }
        atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_lines = [
            "# Workspace Pipeline Report",
            "",
            f"- Workspace root: `{workspace_root}`",
            f"- Status: `{response_status}`",
            f"- Stages: `{len(stages)}`",
            "",
        ]
        for stage in stages:
            markdown_lines.extend(
                [
                    f"## {stage.stage_name}",
                    f"- Status: `{stage.status}`",
                    f"- Summary: {stage.summary}",
                    f"- Artifact: `{stage.artifact_path or 'n/a'}`",
                    "",
                ]
            )
        atomic_write_text(markdown_path, "\n".join(markdown_lines), encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}


workspace_runner_service = WorkspaceRunnerService()
