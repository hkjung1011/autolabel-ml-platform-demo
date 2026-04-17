from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    CommercialMilestone,
    CommercializationPlanRequest,
    CommercializationPlanResponse,
    CommercialStageRequest,
    CommercialStageResponse,
    DatasetDiscoveryRequest,
    SourceCatalogEntry,
    SourceCatalogRequest,
    SourceCatalogResponse,
    StageCandidateRequest,
    WorkspacePipelineRunRequest,
)
from app.services.desktop_package import desktop_package_service
from app.services.program_status import program_status_service
from app.services.reporting import reporting_service
from app.services.research import research_workspace_service
from app.services.review_queue import review_queue_service
from app.services.workspace_runner import workspace_runner_service


class CommercializationService:
    def build_source_catalog(self, request: SourceCatalogRequest) -> SourceCatalogResponse:
        discovery = research_workspace_service.discover_candidates(
            DatasetDiscoveryRequest(
                scan_root=request.scan_root,
                limit=max(1, request.limit),
                min_images=max(1, request.min_images),
            )
        )
        selected_roots = {item.strip() for item in request.selected_dataset_roots if item.strip()}
        candidates = discovery.candidates
        if selected_roots:
            candidates = [candidate for candidate in candidates if candidate.dataset_root in selected_roots]

        entries = [
            SourceCatalogEntry(
                dataset_root=candidate.dataset_root,
                dataset_name=candidate.dataset_name,
                image_count=candidate.image_count,
                group_count=candidate.group_count,
                labeled_anchor_count=candidate.labeled_anchor_count,
                lux_counts=candidate.lux_counts,
                sample_image_path=candidate.sample_image_path,
                source_policy="read_only",
                ingest_mode="copy_only_stage",
                target_stage_name=self._safe_stage_name(candidate.dataset_name),
                recommendation=self._recommend_candidate(candidate.image_count, candidate.labeled_anchor_count),
                notes=list(candidate.notes) + ["Original files must never be edited in-place."],
            )
            for candidate in candidates
        ]

        report_root = Path(request.workspace_root) / "evaluations" / "commercialization"
        report_root.mkdir(parents=True, exist_ok=True)
        response = SourceCatalogResponse(
            scan_root=request.scan_root,
            workspace_root=request.workspace_root,
            total_entries=len(entries),
            protected_source_count=len(entries),
            source_policy_summary="All external-drive datasets are treated as read-only sources. Every experiment must stage copied samples into the workspace imports area.",
            entries=entries,
            report_json_path=str(report_root / "source_catalog.json"),
            report_markdown_path=str(report_root / "source_catalog.md"),
            message=f"Cataloged {len(entries)} protected source datasets from {request.scan_root}.",
        )
        self._write_source_catalog(response)
        return response

    def load_source_catalog(self, workspace_root: str) -> SourceCatalogResponse:
        report_path = Path(workspace_root) / "evaluations" / "commercialization" / "source_catalog.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing source catalog: {report_path}")
        return SourceCatalogResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def stage_protected_source(self, request: CommercialStageRequest) -> CommercialStageResponse:
        stage_root = self._stage_container_root(Path(request.workspace_root))
        stage_result = research_workspace_service.stage_candidate(
            StageCandidateRequest(
                source_dataset_root=request.source_dataset_root,
                workspace_root=str(stage_root),
                staged_name=request.staged_name,
                max_groups=max(1, request.max_groups),
                prefer_labeled=request.prefer_labeled,
                bootstrap_after_stage=request.bootstrap_after_stage,
            )
        )

        pipeline_report_path: str | None = None
        pipeline_markdown_path: str | None = None
        plan_path: str | None = None
        commercial_stage: str | None = None
        commercial_score: int | None = None

        target_workspace = stage_result.staged_workspace_root or request.workspace_root
        if request.run_pipeline_after_stage and stage_result.staged_workspace_root:
            pipeline_response = workspace_runner_service.run_full_pipeline(
                WorkspacePipelineRunRequest(
                    workspace_root=stage_result.staged_workspace_root,
                    include_training_plan=request.include_training_plan,
                    training_dry_run=request.training_dry_run,
                )
            )
            pipeline_report_path = pipeline_response.report_json_path
            pipeline_markdown_path = pipeline_response.report_markdown_path

        if stage_result.staged_workspace_root:
            source_dataset_path = Path(request.source_dataset_root)
            scan_root = source_dataset_path.parent if source_dataset_path.parent.exists() else Path(source_dataset_path.anchor or r"D:\\")
            plan_response = self.build_plan(
                CommercializationPlanRequest(
                    workspace_root=target_workspace,
                    scan_root=str(scan_root),
                    refresh_source_catalog=True,
                    limit=24,
                    min_images=24,
                )
            )
            plan_path = plan_response.report_json_path
            commercial_stage = plan_response.commercial_stage
            commercial_score = plan_response.commercial_readiness_score

        return CommercialStageResponse(
            workspace_root=str(stage_root),
            source_dataset_root=request.source_dataset_root,
            source_policy="read_only",
            ingest_mode="copy_only_stage",
            staged_dataset_root=stage_result.staged_dataset_root,
            staged_workspace_root=stage_result.staged_workspace_root,
            copied_images=stage_result.copied_images,
            copied_labels=stage_result.copied_labels,
            selected_group_count=stage_result.selected_group_count,
            selected_group_ids=stage_result.selected_group_ids,
            bootstrap_message=stage_result.bootstrap_message,
            pipeline_report_path=pipeline_report_path,
            pipeline_markdown_path=pipeline_markdown_path,
            commercial_plan_path=plan_path,
            commercial_stage=commercial_stage,
            commercial_readiness_score=commercial_score,
            message=f"Staged {stage_result.selected_group_count} protected groups into copied workspace artifacts without touching the original source dataset.",
        )

    def build_plan(self, request: CommercializationPlanRequest) -> CommercializationPlanResponse:
        workspace_path = Path(request.workspace_root)
        if request.refresh_source_catalog:
            source_catalog = self.build_source_catalog(
                SourceCatalogRequest(
                    scan_root=request.scan_root,
                    workspace_root=request.workspace_root,
                    limit=request.limit,
                    min_images=request.min_images,
                )
            )
        else:
            source_catalog = self.load_source_catalog(request.workspace_root)

        scorecard = reporting_service.build_scorecard(request.workspace_root)
        program_status = program_status_service.build_report(request.workspace_root)
        package_plan = desktop_package_service.build_plan(request.workspace_root)
        review_queue = self._safe_review_queue(request.workspace_root)
        reviewed_items = 0 if review_queue is None else sum(
            count for status, count in review_queue.status_counts.items() if status != "pending"
        )
        staged_workspace_count = self._count_staged_workspaces(workspace_path)

        commercial_readiness_score = max(
            0,
            min(
                100,
                round(
                    (scorecard.production_score * 0.45)
                    + (scorecard.field_score * 0.20)
                    + (program_status.execution_readiness_percent * 0.15)
                    + (15 if package_plan.build_ready else 0)
                    + (10 if source_catalog.total_entries else 0)
                    + (5 if review_queue is not None else 0)
                ),
            ),
        )

        commercial_stage = self._commercial_stage(commercial_readiness_score)
        strengths = [
            source_catalog.source_policy_summary,
            f"Current operator UI already exposes research score {scorecard.research_score}, field score {scorecard.field_score}, and production score {scorecard.production_score}.",
            "Desktop packaging already produces a Windows exe." if package_plan.build_ready else "Desktop packaging plan exists but still needs a first successful build.",
            f"Review queue has {reviewed_items} reviewed items and is usable for a human-in-the-loop defect workflow."
            if review_queue is not None
            else "Review queue is not active yet.",
            f"{source_catalog.total_entries} protected source datasets are available for staged copy-only expansion.",
        ]
        risks = list(scorecard.blockers)
        if program_status.segmentation_progress_percent < 60:
            risks.append("Segmentation loop is still incomplete, so the product is not yet mask-accuracy ready.")
        if staged_workspace_count < 2:
            risks.append("Only a small number of staged workspaces are active; external validity is still limited.")

        next_actions = [
            "Use the protected source catalog to stage the ship-defect and coating-defect datasets without touching the originals.",
            "Connect ultralytics and opencv-python so production-facing accuracy claims use real model runs.",
            "Add a small mask gold set or SAM-style bootstrap to unlock segmentation metrics.",
            "Promote the review queue from sample approvals to operator-grade approve/reject/edit workflows.",
            "Package the exe with icon, metadata, and an installer once clean-machine smoke tests pass.",
        ]

        milestones = [
            CommercialMilestone(
                milestone_name="Protected Source Intake",
                status="implemented" if source_catalog.total_entries else "planned",
                progress_percent=100 if source_catalog.total_entries else 25,
                summary="External-drive datasets are cataloged as immutable sources, and all downstream work is expected to happen in staged copies only.",
                next_step="Stage the highest-value ship-defect and coating-defect datasets into workspace imports.",
            ),
            CommercialMilestone(
                milestone_name="Pilot Dataset Expansion",
                status="implemented" if staged_workspace_count >= 2 else "partial",
                progress_percent=min(100, 40 + (staged_workspace_count * 20)),
                summary="Multiple staged workspaces make it possible to compare domain transfer without touching source media.",
                next_step="Increase staged candidates so the commercial benchmark is not tied to a single subset.",
            ),
            CommercialMilestone(
                milestone_name="Operator Review Loop",
                status="implemented" if review_queue is not None else "partial",
                progress_percent=100 if review_queue is not None else 45,
                summary="Auto-label proposals can already flow into a human review queue instead of remaining unmanaged files.",
                next_step="Persist edit actions and feed approved items into the next detector retraining cycle.",
            ),
            CommercialMilestone(
                milestone_name="Desktop Distribution",
                status="implemented" if package_plan.build_ready else "partial",
                progress_percent=100 if package_plan.build_ready else 55,
                summary="The Windows packaging path exists so the research UI can evolve toward an operator desktop app.",
                next_step="Smoke-test the packaged exe on a clean Windows machine and attach installer-grade branding.",
            ),
            CommercialMilestone(
                milestone_name="Commercial Accuracy Readiness",
                status="partial" if scorecard.production_score >= 45 else "planned",
                progress_percent=scorecard.production_score,
                summary="Commercial readiness depends on real trainer integration, segmentation coverage, and larger staged datasets.",
                next_step="Run the full loop on the large external datasets and replace proxy evidence with production metrics.",
            ),
        ]

        report_root = workspace_path / "evaluations" / "commercialization"
        report_root.mkdir(parents=True, exist_ok=True)
        response = CommercializationPlanResponse(
            workspace_root=request.workspace_root,
            scan_root=request.scan_root,
            commercial_stage=commercial_stage,
            commercial_readiness_score=commercial_readiness_score,
            protected_source_count=source_catalog.protected_source_count,
            staged_workspace_count=staged_workspace_count,
            source_catalog_path=source_catalog.report_json_path,
            research_score=scorecard.research_score,
            field_score=scorecard.field_score,
            production_score=scorecard.production_score,
            strengths=self._dedupe(strengths),
            risks=self._dedupe(risks),
            next_actions=self._dedupe(next_actions),
            milestones=milestones,
            report_json_path=str(report_root / "commercial_plan.json"),
            report_markdown_path=str(report_root / "commercial_plan.md"),
        )
        self._write_plan(response)
        return response

    def load_plan(self, workspace_root: str) -> CommercializationPlanResponse:
        report_path = Path(workspace_root) / "evaluations" / "commercialization" / "commercial_plan.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing commercialization plan: {report_path}")
        return CommercializationPlanResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _safe_review_queue(self, workspace_root: str):
        try:
            return review_queue_service.load_queue(workspace_root)
        except FileNotFoundError:
            return None

    def _count_staged_workspaces(self, workspace_path: Path) -> int:
        if workspace_path.name == "candidate_workspaces":
            root = workspace_path
        elif workspace_path.parent.name == "candidate_workspaces":
            root = workspace_path.parent
        else:
            root = workspace_path / "candidate_workspaces"
        if not root.exists():
            return 0
        return sum(1 for item in root.iterdir() if item.is_dir())

    def _stage_container_root(self, workspace_path: Path) -> Path:
        if workspace_path.parent.name == "candidate_workspaces":
            return workspace_path.parent.parent
        if workspace_path.name == "candidate_workspaces":
            return workspace_path.parent
        return workspace_path

    def _commercial_stage(self, score: int) -> str:
        if score >= 80:
            return "commercializing"
        if score >= 65:
            return "pilot-ready"
        if score >= 50:
            return "pilot-prep"
        return "research-to-pilot"

    def _recommend_candidate(self, image_count: int, labeled_anchor_count: int) -> str:
        if labeled_anchor_count >= 50:
            return "High-priority pilot candidate"
        if labeled_anchor_count >= 10:
            return "Good staged validation candidate"
        if image_count >= 500:
            return "Large source set, but labeling is still needed"
        return "Keep as protected source until labeling strategy is decided"

    def _safe_stage_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value).strip("._-") or "candidate"

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

    def _write_source_catalog(self, response: SourceCatalogResponse) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
        )
        lines = [
            "# Protected Source Catalog",
            "",
            f"- Scan root: `{response.scan_root}`",
            f"- Workspace root: `{response.workspace_root}`",
            f"- Entries: **{response.total_entries}**",
            f"- Protected sources: **{response.protected_source_count}**",
            "",
            f"- Policy: {response.source_policy_summary}",
            "",
            "## Entries",
        ]
        for entry in response.entries:
            lines.append(
                f"- `{entry.dataset_name}` images={entry.image_count}, groups={entry.group_count}, labels={entry.labeled_anchor_count}, policy={entry.source_policy}, ingest={entry.ingest_mode}, stage={entry.target_stage_name}"
            )
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n")

    def _write_plan(self, response: CommercializationPlanResponse) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
        )
        lines = [
            "# Commercialization Plan",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Scan root: `{response.scan_root}`",
            f"- Commercial stage: **{response.commercial_stage}**",
            f"- Commercial readiness score: **{response.commercial_readiness_score}**",
            f"- Protected sources: **{response.protected_source_count}**",
            f"- Staged workspaces: **{response.staged_workspace_count}**",
            f"- Source catalog: `{response.source_catalog_path}`",
            "",
            "## Scores",
            f"- Research: {response.research_score}",
            f"- Field: {response.field_score}",
            f"- Production: {response.production_score}",
            "",
            "## Strengths",
        ]
        for item in response.strengths:
            lines.append(f"- {item}")
        lines.extend(["", "## Risks"])
        for item in response.risks:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in response.next_actions:
            lines.append(f"- {item}")
        lines.extend(["", "## Milestones"])
        for item in response.milestones:
            lines.append(
                f"- `{item.milestone_name}` status={item.status}, progress={item.progress_percent}%"
            )
            lines.append(f"  - summary: {item.summary}")
            lines.append(f"  - next: {item.next_step}")
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n")


commercialization_service = CommercializationService()
