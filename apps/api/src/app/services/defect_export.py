from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.domain.defect_autolabel_models import (
    CANONICAL_CLASS_NAMES,
    DOMAIN_ALLOWED_CLASS_IDS,
    DefectAssetRecord,
    DefectExportRequest,
    DefectExportResponse,
    DefectReviewItem,
)


class DefectExportService:
    def export_dataset(self, request: DefectExportRequest) -> DefectExportResponse:
        workspace_path = Path(request.workspace_root)
        assets = self._load_assets(workspace_path)
        items = self._load_review_items(workspace_path)
        export_root = workspace_path / "exports" / "yolo_detection"
        if request.overwrite and export_root.exists():
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

        export_class_ids = DOMAIN_ALLOWED_CLASS_IDS[request.domain]
        export_id_map = {class_id: index for index, class_id in enumerate(export_class_ids)}
        accepted_statuses = {"auto_accepted", "approved"}
        accepted_items_by_asset: dict[str, list[DefectReviewItem]] = {}
        for item in items:
            if item.review_status not in accepted_statuses:
                continue
            accepted_items_by_asset.setdefault(item.asset_id, []).append(item)

        exported_images = 0
        exported_labels = 0
        accepted_proposals = 0
        export_manifest: list[dict[str, object]] = []
        for asset in assets:
            dest_image = export_root / "images" / asset.split / Path(asset.image_path).name
            dest_label = export_root / "labels" / asset.split / f"{Path(asset.image_path).stem}.txt"
            shutil.copy2(asset.image_path, dest_image)
            exported_images += 1

            lines = []
            for item in accepted_items_by_asset.get(asset.asset_id, []):
                if item.class_id not in export_id_map:
                    continue
                export_class_id = export_id_map[item.class_id]
                x_center, y_center, box_width, box_height = item.bbox_yolo
                lines.append(f"{export_class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                accepted_proposals += 1
            atomic_write_text(dest_label, "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            exported_labels += 1
            export_manifest.append(
                {
                    "asset_id": asset.asset_id,
                    "original_image_path": asset.image_path,
                    "export_image_path": str(dest_image),
                    "export_label_path": str(dest_label),
                    "accepted_proposal_count": len(lines),
                }
            )

        dataset_yaml_path = export_root / "meta" / "dataset.yaml"
        yaml_lines = [
            f"path: {export_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(export_class_ids)}",
            "names:",
        ]
        for class_id, export_index in export_id_map.items():
            yaml_lines.append(f"  {export_index}: {CANONICAL_CLASS_NAMES[class_id]}")
        atomic_write_text(dataset_yaml_path, "\n".join(yaml_lines) + "\n", encoding="utf-8")

        report_json_path = workspace_path / "reports" / "export_summary.json"
        report_markdown_path = workspace_path / "reports" / "export_summary.md"
        atomic_write_json(export_root / "meta" / "export_manifest.json", export_manifest)
        response = DefectExportResponse(
            workspace_root=str(workspace_path),
            domain=request.domain,
            export_root=str(export_root),
            dataset_yaml_path=str(dataset_yaml_path),
            exported_images=exported_images,
            exported_labels=exported_labels,
            accepted_proposals=accepted_proposals,
            report_json_path=str(report_json_path),
            report_markdown_path=str(report_markdown_path),
            message="Reviewed original-image defect labels exported in YOLO detection format.",
        )
        atomic_write_json(report_json_path, response.model_dump())
        lines = [
            "# Defect Export Summary",
            "",
            f"- Workspace: `{workspace_path}`",
            f"- Export root: `{export_root}`",
            f"- Exported images: **{exported_images}**",
            f"- Exported labels: **{exported_labels}**",
            f"- Accepted proposals: **{accepted_proposals}**",
            f"- Dataset yaml: `{dataset_yaml_path}`",
        ]
        atomic_write_text(report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")
        return response

    def _load_assets(self, workspace_path: Path) -> list[DefectAssetRecord]:
        manifest_path = workspace_path / "manifests" / "asset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing asset manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [DefectAssetRecord.model_validate(item) for item in payload]

    def _load_review_items(self, workspace_path: Path) -> list[DefectReviewItem]:
        items_path = workspace_path / "review_queue" / "items.json"
        if not items_path.exists():
            raise FileNotFoundError(f"Missing review queue items: {items_path}")
        payload = json.loads(items_path.read_text(encoding="utf-8"))
        return [DefectReviewItem.model_validate(item) for item in payload]

    def _safe_rmtree(self, target: Path, workspace_root: Path) -> None:
        resolved_target = target.resolve()
        resolved_workspace = workspace_root.resolve()
        if resolved_workspace not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_target}")
        shutil.rmtree(resolved_target, ignore_errors=False)


defect_export_service = DefectExportService()
