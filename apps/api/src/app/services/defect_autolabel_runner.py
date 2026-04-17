from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.domain.defect_autolabel_models import (
    DefectAutolabelProjectRequest,
    DefectAutolabelProjectResponse,
    DefectAutolabelRunRequest,
    DefectAutolabelRunResponse,
    DefectProposal,
)
from app.services.defect_anchor_selector import defect_anchor_selector_service
from app.services.defect_dataset_ingest import defect_dataset_ingest_service
from app.services.defect_inference import defect_inference_service
from app.services.defect_propagation import defect_propagation_service
from app.services.defect_review import defect_review_service


class DefectAutolabelRunnerService:
    def init_project(self, request: DefectAutolabelProjectRequest) -> DefectAutolabelProjectResponse:
        return defect_dataset_ingest_service.init_project(request)

    def run(self, request: DefectAutolabelRunRequest) -> DefectAutolabelRunResponse:
        workspace_path = Path(request.workspace_root)
        if request.overwrite:
            self._reset_outputs(workspace_path)

        assets = defect_dataset_ingest_service.load_assets(workspace_path)
        groups = defect_dataset_ingest_service.load_groups(workspace_path)
        quality_by_asset_id = defect_dataset_ingest_service.load_quality_map(workspace_path)
        assets_by_id = {asset.asset_id: asset for asset in assets}

        groups = defect_anchor_selector_service.select_anchors(
            groups=groups,
            assets_by_id=assets_by_id,
            quality_by_asset_id=quality_by_asset_id,
        )
        defect_dataset_ingest_service.save_groups(workspace_path, groups)

        direct_proposals_by_asset: dict[str, list[DefectProposal]]
        if request.run_mode in {"full", "detect_only"}:
            direct_proposals_by_asset = {}
            for asset in assets:
                direct = defect_inference_service.detect_asset(
                    asset=asset,
                    quality=quality_by_asset_id.get(asset.asset_id, {}),
                    domain=request.domain,
                )
                direct_proposals_by_asset[asset.asset_id] = direct
        else:
            direct_proposals_by_asset = self._load_manifest_proposals(workspace_path / "predictions" / "raw_manifest.json")

        propagation_reports: list[dict[str, object]] = []
        fused_proposals_by_asset = {asset_id: [proposal.model_copy() for proposal in proposals] for asset_id, proposals in direct_proposals_by_asset.items()}
        if request.run_mode in {"full", "propagate_only"}:
            propagated_map, propagation_reports = defect_propagation_service.propagate_groups(
                groups=groups,
                assets_by_id=assets_by_id,
                proposals_by_asset_id=direct_proposals_by_asset,
            )
            fused_proposals_by_asset = {}
            for asset in assets:
                proposals = propagated_map.get(asset.asset_id, [])
                direct = [proposal for proposal in proposals if proposal.source_mode == "direct_detect"]
                propagated = [proposal for proposal in proposals if proposal.source_mode != "direct_detect"]
                if propagated:
                    fused = defect_inference_service.merge_direct_and_propagated(
                        direct_proposals=direct,
                        propagated_proposals=propagated,
                        quality=quality_by_asset_id.get(asset.asset_id, {}),
                        split=asset.split,
                    )
                else:
                    fused = direct
                fused_proposals_by_asset[asset.asset_id] = fused

        raw_manifest = []
        fused_manifest = []
        for asset in assets:
            raw_predictions = direct_proposals_by_asset.get(asset.asset_id, [])
            fused_predictions = fused_proposals_by_asset.get(asset.asset_id, raw_predictions)
            atomic_write_json(workspace_path / "predictions" / "raw" / f"{asset.asset_id}.json", [item.model_dump() for item in raw_predictions])
            atomic_write_json(workspace_path / "predictions" / "fused" / f"{asset.asset_id}.json", [item.model_dump() for item in fused_predictions])
            raw_manifest.extend(item.model_dump() for item in raw_predictions)
            fused_manifest.extend(item.model_dump() for item in fused_predictions)

        raw_manifest_path = workspace_path / "predictions" / "raw_manifest.json"
        fused_manifest_path = workspace_path / "predictions" / "fused_manifest.json"
        propagation_report_path = workspace_path / "reports" / "propagation_summary.json"
        atomic_write_json(raw_manifest_path, raw_manifest)
        atomic_write_json(fused_manifest_path, fused_manifest)
        atomic_write_json(propagation_report_path, propagation_reports)

        review_response = defect_review_service.build_queue(workspace_path)
        report_json_path = workspace_path / "reports" / "defect_autolabel_run.json"
        report_markdown_path = workspace_path / "reports" / "defect_autolabel_run.md"
        response = DefectAutolabelRunResponse(
            workspace_root=str(workspace_path),
            domain=request.domain,
            total_assets=len(assets),
            grouped_assets=sum(1 for group in groups if len(group.member_asset_ids) > 1),
            total_groups=len(groups),
            anchor_selected_groups=sum(1 for group in groups if group.anchor_asset_id is not None),
            proposal_count=len(fused_manifest),
            review_required_count=sum(1 for item in fused_manifest if item["review_required"]),
            manifest_path=str(fused_manifest_path),
            review_queue_path=review_response.items_json_path,
            report_json_path=str(report_json_path),
            report_markdown_path=str(report_markdown_path),
            message="Defect auto-label run completed on original images.",
        )
        atomic_write_json(report_json_path, response.model_dump())
        lines = [
            "# Defect Auto-Label Run",
            "",
            f"- Workspace: `{workspace_path}`",
            f"- Domain: `{request.domain}`",
            f"- Total assets: **{len(assets)}**",
            f"- Paired groups: **{response.grouped_assets}**",
            f"- Anchor-selected groups: **{response.anchor_selected_groups}**",
            f"- Total fused proposals: **{response.proposal_count}**",
            f"- Review-required proposals: **{response.review_required_count}**",
            f"- Raw manifest: `{raw_manifest_path}`",
            f"- Fused manifest: `{fused_manifest_path}`",
            f"- Review queue: `{review_response.items_json_path}`",
        ]
        atomic_write_text(report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")
        return response

    def load_latest(self, workspace_root: str) -> DefectAutolabelRunResponse:
        report_path = Path(workspace_root) / "reports" / "defect_autolabel_run.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing V2 run report: {report_path}")
        return DefectAutolabelRunResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _load_manifest_proposals(self, manifest_path: Path) -> dict[str, list[DefectProposal]]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing prediction manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        proposals = [DefectProposal.model_validate(item) for item in payload]
        grouped: dict[str, list[DefectProposal]] = {}
        for proposal in proposals:
            grouped.setdefault(proposal.asset_id, []).append(proposal)
        return grouped

    def _reset_outputs(self, workspace_path: Path) -> None:
        for target in [
            workspace_path / "predictions",
            workspace_path / "review_queue",
            workspace_path / "exports" / "yolo_detection",
        ]:
            if not target.exists():
                continue
            self._safe_rmtree(target, workspace_path)

    def _safe_rmtree(self, target: Path, workspace_path: Path) -> None:
        resolved_target = target.resolve()
        resolved_workspace = workspace_path.resolve()
        if resolved_workspace not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_target}")
        shutil.rmtree(resolved_target, ignore_errors=False)


defect_autolabel_runner_service = DefectAutolabelRunnerService()
