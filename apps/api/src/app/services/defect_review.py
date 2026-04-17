from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.domain.defect_autolabel_models import (
    DefectProposal,
    DefectReviewBuildResponse,
    DefectReviewItem,
    DefectReviewUpdateRequest,
)


class DefectReviewService:
    def build_queue(self, workspace_root: str | Path) -> DefectReviewBuildResponse:
        workspace_path = Path(workspace_root)
        proposals = self._load_proposals(workspace_path)
        existing = self._load_existing_map(workspace_path)
        items: list[DefectReviewItem] = []
        for proposal in proposals:
            current = existing.get(proposal.proposal_id)
            review_status = current.review_status if current else ("pending" if proposal.review_required else "auto_accepted")
            notes = list(current.notes) if current else self._default_notes(proposal)
            item = DefectReviewItem(
                proposal_id=proposal.proposal_id,
                asset_id=proposal.asset_id,
                image_path=proposal.image_path,
                split=proposal.split,
                image_width=proposal.image_width,
                image_height=proposal.image_height,
                class_id=proposal.class_id,
                class_name=proposal.class_name,
                confidence=proposal.confidence,
                bbox_xyxy=list(proposal.bbox_xyxy),
                bbox_yolo=list(proposal.bbox_yolo),
                source_mode=proposal.source_mode,
                views_supporting=list(proposal.views_supporting),
                lux_bucket=proposal.lux_bucket,
                priority=proposal.priority,
                quality_flags=list(proposal.quality_flags),
                review_status=review_status,
                review_owner=current.review_owner if current else None,
                updated_at=current.updated_at if current else None,
                notes=notes,
                review_history=list(current.review_history) if current else [],
            )
            items.append(item)
        return self._write_response(workspace_path=workspace_path, items=items, message="Defect auto-label review queue built.")

    def update_queue_item(self, request: DefectReviewUpdateRequest) -> DefectReviewBuildResponse:
        workspace_path = Path(request.workspace_root)
        items = self._load_items(workspace_path)
        updated = False
        new_status = {
            "approve": "approved",
            "reject": "rejected",
            "needs_edit": "needs_edit",
            "reset": "pending",
        }[request.action]
        for item in items:
            if item.proposal_id != request.proposal_id:
                continue
            item.review_status = new_status
            owner = (request.review_owner or "").strip()
            note = (request.note or "").strip()
            if owner:
                item.review_owner = owner
            if note:
                item.notes.append(note)
            timestamp = datetime.now().isoformat(timespec="seconds")
            item.updated_at = timestamp
            history = f"{timestamp} :: {request.action}"
            if owner:
                history += f" by {owner}"
            if note:
                history += f" :: {note}"
            item.review_history.append(history)
            updated = True
            break
        if not updated:
            raise FileNotFoundError(f"Proposal not found in review queue: {request.proposal_id}")
        return self._write_response(workspace_path=workspace_path, items=items, message=f"Review item `{request.proposal_id}` updated.")

    def _load_proposals(self, workspace_path: Path) -> list[DefectProposal]:
        predictions_path = workspace_path / "predictions" / "fused_manifest.json"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing fused prediction manifest: {predictions_path}")
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
        return [DefectProposal.model_validate(item) for item in payload]

    def _load_existing_map(self, workspace_path: Path) -> dict[str, DefectReviewItem]:
        try:
            return {item.proposal_id: item for item in self._load_items(workspace_path)}
        except FileNotFoundError:
            return {}

    def _load_items(self, workspace_path: Path) -> list[DefectReviewItem]:
        items_path = workspace_path / "review_queue" / "items.json"
        if not items_path.exists():
            raise FileNotFoundError(f"Missing review queue items: {items_path}")
        payload = json.loads(items_path.read_text(encoding="utf-8"))
        return [DefectReviewItem.model_validate(item) for item in payload]

    def _write_response(
        self,
        *,
        workspace_path: Path,
        items: list[DefectReviewItem],
        message: str,
    ) -> DefectReviewBuildResponse:
        status_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for item in items:
            status_counts[item.review_status] = status_counts.get(item.review_status, 0) + 1
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1

        items_path = workspace_path / "review_queue" / "items.json"
        report_json_path = workspace_path / "reports" / "review_summary.json"
        report_markdown_path = workspace_path / "reports" / "review_summary.md"
        response = DefectReviewBuildResponse(
            workspace_root=str(workspace_path),
            total_items=len(items),
            reviewed_count=sum(1 for item in items if item.review_status not in {"pending", "auto_accepted"}),
            status_counts=status_counts,
            priority_counts=priority_counts,
            items=items[:60],
            items_json_path=str(items_path),
            report_json_path=str(report_json_path),
            report_markdown_path=str(report_markdown_path),
            message=message,
        )
        atomic_write_json(items_path, [item.model_dump() for item in items])
        atomic_write_json(report_json_path, response.model_dump())
        lines = [
            "# Defect Review Queue",
            "",
            f"- Workspace: `{workspace_path}`",
            f"- Total items: **{response.total_items}**",
            f"- Reviewed items: **{response.reviewed_count}**",
            "",
            "## Status counts",
        ]
        for key, value in sorted(response.status_counts.items()):
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Priority counts"])
        for key, value in sorted(response.priority_counts.items()):
            lines.append(f"- `{key}`: {value}")
        atomic_write_text(report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")
        return response

    def _default_notes(self, proposal: DefectProposal) -> list[str]:
        notes: list[str] = []
        if proposal.lux_bucket in {"very_dark", "glare"}:
            notes.append("Lighting risk detected; confirm label before training export.")
        if proposal.source_mode != "direct_detect":
            notes.append("Propagated label; confirm alignment before approval.")
        if "boundary_touching" in proposal.quality_flags:
            notes.append("Boundary-touching box; verify truncation is acceptable.")
        return notes or ["Auto-accepted proposal unless reviewer marks otherwise."]


defect_review_service = DefectReviewService()
