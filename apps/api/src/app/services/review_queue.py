from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    ReviewQueueExportResponse,
    ReviewQueueItem,
    ReviewQueueResponse,
    ReviewQueueUpdateRequest,
)


class ReviewQueueService:
    def build_queue(self, workspace_root: str) -> ReviewQueueResponse:
        workspace_path = Path(workspace_root)
        autolabel_root = workspace_path / "evaluations" / "autolabel"
        bootstrap_report_path = autolabel_root / "bootstrap_report.json"
        proposals_path = autolabel_root / "proposals.json"
        if not bootstrap_report_path.exists() or not proposals_path.exists():
            raise FileNotFoundError("Auto-label bootstrap report is missing. Build auto-label bootstrap first.")

        proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
        previous_items = self._load_existing_items_map(workspace_path)
        items: list[ReviewQueueItem] = []

        for proposal in proposals:
            split = proposal.get("split", "train")
            arm_name = proposal.get("arm_name", "unknown")
            proposal_mode = proposal.get("proposal_mode", "label_reuse")
            priority = self._priority_for(split=split, arm_name=arm_name, proposal_mode=proposal_mode)
            proposal_id = self._proposal_id_for(proposal)
            existing = previous_items.get(proposal_id)
            review_status = existing.review_status if existing else "pending"
            notes = list(existing.notes) if existing else self._notes_for(split=split, arm_name=arm_name, proposal_mode=proposal_mode)
            review_history = list(existing.review_history) if existing else []
            item = ReviewQueueItem(
                proposal_id=proposal_id,
                arm_name=arm_name,
                split=split,
                proposal_mode=proposal_mode,
                review_status=review_status,
                priority=priority,
                source_image_path=proposal.get("source_image_path", ""),
                source_label_path=proposal.get("source_label_path", ""),
                image_path=proposal.get("output_image_path", ""),
                label_path=proposal.get("output_label_path", ""),
                review_owner=existing.review_owner if existing else None,
                updated_at=existing.updated_at if existing else None,
                notes=notes,
                review_history=review_history,
            )
            items.append(item)

        response = self._make_response(
            workspace_path=workspace_path,
            items=items,
            message="Review queue built from the auto-label bootstrap proposals.",
        )
        self._write_report(response, items)
        return response

    def load_queue(self, workspace_root: str) -> ReviewQueueResponse:
        report_path = Path(workspace_root) / "evaluations" / "review_queue" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing review queue report: {report_path}")
        return ReviewQueueResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def update_queue_item(self, request: ReviewQueueUpdateRequest) -> ReviewQueueResponse:
        workspace_path = Path(request.workspace_root)
        items = self._load_all_items(workspace_path)
        if request.action not in {"approve", "reject", "needs_edit", "reset"}:
            raise ValueError(f"Unsupported review action: {request.action}")
        new_status = {
            "approve": "approved",
            "reject": "rejected",
            "needs_edit": "needs_edit",
            "reset": "pending",
        }[request.action]

        updated = False
        for item in items:
            if item.proposal_id != request.proposal_id:
                continue
            item.review_status = new_status
            owner = (request.review_owner or "").strip()
            if owner:
                item.review_owner = owner
            note = (request.note or "").strip()
            if note:
                item.notes.append(note)
            timestamp = datetime.now().isoformat(timespec="seconds")
            item.updated_at = timestamp
            history_note = f"{timestamp} :: {request.action}"
            if owner:
                history_note += f" by {owner}"
            if note:
                history_note += f" :: {note}"
            item.review_history.append(history_note)
            updated = True
            break
        if not updated:
            raise FileNotFoundError(f"Proposal not found in review queue: {request.proposal_id}")

        response = self._make_response(
            workspace_path=workspace_path,
            items=items,
            message=f"Review queue item `{request.proposal_id}` updated to `{new_status}`.",
        )
        self._write_report(response, items)
        return response

    def export_approved_dataset(self, workspace_root: str) -> ReviewQueueExportResponse:
        workspace_path = Path(workspace_root)
        items = self._load_all_items(workspace_path)
        approved_items = [item for item in items if item.review_status == "approved"]
        export_root = workspace_path / "datasets" / "autolabel" / "approved_reviewed"
        if export_root.exists():
            self._safe_rmtree(export_root, workspace_path)
        for relative in [
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
            "meta",
        ]:
            (export_root / relative).mkdir(parents=True, exist_ok=True)

        split_counts = {"train": 0, "val": 0, "test": 0}
        exported_manifest: list[dict[str, str]] = []
        for item in approved_items:
            image_path = Path(item.image_path)
            label_path = Path(item.label_path)
            if not image_path.exists() or not label_path.exists():
                continue
            dest_image = export_root / "images" / item.split / image_path.name
            dest_label = export_root / "labels" / item.split / label_path.name
            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)
            split_counts[item.split] = split_counts.get(item.split, 0) + 1
            exported_manifest.append(
                {
                    "proposal_id": item.proposal_id,
                    "review_owner": item.review_owner or "",
                    "split": item.split,
                    "arm_name": item.arm_name,
                    "image_path": str(dest_image),
                    "label_path": str(dest_label),
                }
            )

        report_root = workspace_path / "evaluations" / "review_queue"
        report_root.mkdir(parents=True, exist_ok=True)
        response = ReviewQueueExportResponse(
            workspace_root=str(workspace_path),
            approved_dataset_root=str(export_root),
            exported_items=len(exported_manifest),
            split_counts=split_counts,
            items_json_path=str(report_root / "approved_export_items.json"),
            report_json_path=str(report_root / "approved_export_report.json"),
            report_markdown_path=str(report_root / "approved_export_report.md"),
            message=f"Exported {len(exported_manifest)} approved review items into a detector-training dataset.",
        )
        atomic_write_text(response.items_json_path, json.dumps(exported_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        atomic_write_text(response.report_json_path, json.dumps(response.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# Approved Review Export",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Approved dataset root: `{response.approved_dataset_root}`",
            f"- Exported items: **{response.exported_items}**",
            "",
            "## Split Counts",
        ]
        for split, count in response.split_counts.items():
            lines.append(f"- `{split}`: {count}")
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")

        refreshed = self._make_response(
            workspace_path=workspace_path,
            items=items,
            message=response.message,
        )
        self._write_report(refreshed, items)
        return response

    def _priority_for(self, *, split: str, arm_name: str, proposal_mode: str) -> str:
        if split in {"val", "test"}:
            return "high"
        if arm_name in {"retinex", "daf"} or proposal_mode != "anchor_label_reuse":
            return "medium"
        return "normal"

    def _notes_for(self, *, split: str, arm_name: str, proposal_mode: str) -> list[str]:
        notes = []
        if split in {"val", "test"}:
            notes.append("Evaluation split item; review before using in accuracy claims.")
        if arm_name in {"retinex", "mertens", "daf"}:
            notes.append("Transformed variant; verify label reuse safety.")
        if proposal_mode == "registered_variant_reuse":
            notes.append("Registered variant reuse path; inspect alignment-sensitive defects.")
        if proposal_mode == "fusion_variant_reuse":
            notes.append("Fusion branch proposal; inspect whether defect boundaries stayed intact.")
        return notes or ["Seed review item."]

    def _load_all_items(self, workspace_path: Path) -> list[ReviewQueueItem]:
        items_path = workspace_path / "evaluations" / "review_queue" / "items.json"
        if not items_path.exists():
            raise FileNotFoundError(f"Missing review queue items: {items_path}")
        payload = json.loads(items_path.read_text(encoding="utf-8"))
        return [ReviewQueueItem.model_validate(item) for item in payload]

    def _load_existing_items_map(self, workspace_path: Path) -> dict[str, ReviewQueueItem]:
        try:
            return {item.proposal_id: item for item in self._load_all_items(workspace_path)}
        except FileNotFoundError:
            return {}

    def _proposal_id_for(self, proposal: dict[str, str]) -> str:
        token = "::".join(
            [
                proposal.get("arm_name", ""),
                proposal.get("split", ""),
                proposal.get("proposal_mode", ""),
                proposal.get("output_image_path", ""),
                proposal.get("output_label_path", ""),
            ]
        )
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        return f"proposal_{digest}"

    def _make_response(self, *, workspace_path: Path, items: list[ReviewQueueItem], message: str) -> ReviewQueueResponse:
        status_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        arm_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for item in items:
            status_counts[item.review_status] = status_counts.get(item.review_status, 0) + 1
            split_counts[item.split] = split_counts.get(item.split, 0) + 1
            arm_counts[item.arm_name] = arm_counts.get(item.arm_name, 0) + 1
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1

        report_root = workspace_path / "evaluations" / "review_queue"
        report_root.mkdir(parents=True, exist_ok=True)
        approved_export_path = workspace_path / "datasets" / "autolabel" / "approved_reviewed"
        return ReviewQueueResponse(
            workspace_root=str(workspace_path),
            total_items=len(items),
            reviewed_count=sum(1 for item in items if item.review_status != "pending"),
            status_counts=status_counts,
            split_counts=split_counts,
            arm_counts=arm_counts,
            priority_counts=priority_counts,
            items=items[:60],
            items_json_path=str(report_root / "items.json"),
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
            approved_dataset_root=str(approved_export_path) if approved_export_path.exists() else None,
            approved_export_count=sum(1 for item in items if item.review_status == "approved"),
            message=message,
        )

    def _write_report(self, response: ReviewQueueResponse, items: list[ReviewQueueItem]) -> None:
        atomic_write_text(response.report_json_path, json.dumps(response.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        atomic_write_text(
            response.items_json_path,
            json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Review Queue",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Total items: **{response.total_items}**",
            f"- Reviewed items: **{response.reviewed_count}**",
            f"- Approved dataset root: `{response.approved_dataset_root or 'n/a'}`",
            "",
            "## Status Counts",
        ]
        for key, value in response.status_counts.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Priority Counts"])
        for key, value in response.priority_counts.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Split Counts"])
        for key, value in response.split_counts.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Arm Counts"])
        for key, value in response.arm_counts.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Sample Items"])
        for item in response.items[:20]:
            lines.append(
                f"- `{item.proposal_id}` arm={item.arm_name}, split={item.split}, priority={item.priority}, status={item.review_status}, mode={item.proposal_mode}, owner={item.review_owner or 'n/a'}"
            )
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")

    def _safe_rmtree(self, target: Path, workspace_path: Path) -> None:
        resolved_target = target.resolve()
        resolved_workspace = workspace_path.resolve()
        if resolved_workspace not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_target}")
        shutil.rmtree(resolved_target, ignore_errors=False)


review_queue_service = ReviewQueueService()
