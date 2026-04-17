from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.domain.defect_autolabel_models import (
    DefectAssetRecord,
    DefectAutolabelProjectRequest,
    DefectAutolabelProjectResponse,
    DefectGroupRecord,
)
from app.services.defect_quality import defect_quality_service


class DefectDatasetIngestService:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    LUX_PATTERN = re.compile(r"(?i)(?:^|[_-])lux[_-]?(\d+)")
    SHOT_PATTERN = re.compile(r"(?i)(shot[_-]?\d+)")

    def init_project(self, request: DefectAutolabelProjectRequest) -> DefectAutolabelProjectResponse:
        input_root = Path(request.input_root)
        workspace_root = Path(request.workspace_root)
        if not input_root.exists():
            raise FileNotFoundError(f"Input root not found: {input_root}")

        asset_records: list[DefectAssetRecord] = []
        quality_records: list[dict[str, object]] = []
        groups: dict[str, DefectGroupRecord] = {}

        for image_path in sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES):
            width, height = Image.open(image_path).size
            asset_id = self._asset_id_for(input_root=input_root, image_path=image_path)
            reported_lux = self._reported_lux_for(image_path)
            base_group_id = self._group_id_for(image_path)
            group_id = base_group_id if request.dataset_mode == "paired_lux" else asset_id
            quality = defect_quality_service.build_metrics(asset_id=asset_id, image_path=image_path)
            asset = DefectAssetRecord(
                asset_id=asset_id,
                image_path=str(image_path),
                group_id=group_id,
                shot_id=self._shot_id_for(image_path),
                reported_lux=reported_lux,
                estimated_lux_bucket=quality.lux_bucket,
                split=self._split_for(image_path),
                width=width,
                height=height,
            )
            asset_records.append(asset)
            quality_records.append(quality.model_dump())

            group = groups.setdefault(
                group_id,
                DefectGroupRecord(
                    group_id=group_id,
                    member_asset_ids=[],
                    anchor_asset_id=None,
                    group_mode="paired_lux" if request.dataset_mode == "paired_lux" else "single_image",
                    available_luxes=[],
                ),
            )
            group.member_asset_ids.append(asset_id)
            if reported_lux is not None and reported_lux not in group.available_luxes:
                group.available_luxes.append(reported_lux)

            atomic_write_json(workspace_root / "analyses" / "quality" / f"{asset_id}.json", quality.model_dump())

        if not asset_records:
            raise FileNotFoundError(f"No supported images found under: {input_root}")

        group_records = []
        for group in groups.values():
            group.available_luxes = sorted(group.available_luxes)
            group_records.append(group)

        asset_manifest_path = workspace_root / "manifests" / "asset_manifest.json"
        group_manifest_path = workspace_root / "manifests" / "group_manifest.json"
        quality_manifest_path = workspace_root / "manifests" / "quality_manifest.json"
        report_json_path = workspace_root / "reports" / "defect_project_init.json"
        report_markdown_path = workspace_root / "reports" / "defect_project_init.md"

        atomic_write_json(asset_manifest_path, [asset.model_dump() for asset in asset_records])
        atomic_write_json(group_manifest_path, [group.model_dump() for group in group_records])
        atomic_write_json(quality_manifest_path, quality_records)

        response = DefectAutolabelProjectResponse(
            workspace_root=str(workspace_root),
            domain=request.domain,
            total_assets=len(asset_records),
            total_groups=len(group_records),
            asset_manifest_path=str(asset_manifest_path),
            group_manifest_path=str(group_manifest_path),
            quality_manifest_path=str(quality_manifest_path),
            report_json_path=str(report_json_path),
            report_markdown_path=str(report_markdown_path),
            message="Defect auto-label project initialized from original images.",
        )
        atomic_write_json(report_json_path, response.model_dump())
        lines = [
            "# Defect Auto-Label Project Init",
            "",
            f"- Input root: `{input_root}`",
            f"- Workspace root: `{workspace_root}`",
            f"- Domain: `{request.domain}`",
            f"- Dataset mode: `{request.dataset_mode}`",
            f"- Total assets: **{len(asset_records)}**",
            f"- Total groups: **{len(group_records)}**",
            "",
            "## Sample groups",
        ]
        for group in group_records[:12]:
            lines.append(
                f"- `{group.group_id}` assets={len(group.member_asset_ids)}, luxes={group.available_luxes or ['n/a']}, mode={group.group_mode}"
            )
        atomic_write_text(report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")
        return response

    def load_assets(self, workspace_root: str | Path) -> list[DefectAssetRecord]:
        manifest_path = Path(workspace_root) / "manifests" / "asset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing asset manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [DefectAssetRecord.model_validate(item) for item in payload]

    def load_groups(self, workspace_root: str | Path) -> list[DefectGroupRecord]:
        manifest_path = Path(workspace_root) / "manifests" / "group_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing group manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [DefectGroupRecord.model_validate(item) for item in payload]

    def load_quality_map(self, workspace_root: str | Path) -> dict[str, dict[str, object]]:
        manifest_path = Path(workspace_root) / "manifests" / "quality_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing quality manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {item["asset_id"]: item for item in payload}

    def save_groups(self, workspace_root: str | Path, groups: list[DefectGroupRecord]) -> None:
        atomic_write_json(Path(workspace_root) / "manifests" / "group_manifest.json", [group.model_dump() for group in groups])

    def _asset_id_for(self, *, input_root: Path, image_path: Path) -> str:
        try:
            token = str(image_path.relative_to(input_root))
        except ValueError:
            token = str(image_path)
        return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]

    def _reported_lux_for(self, image_path: Path) -> int | None:
        candidates = [image_path.stem, *[part for part in image_path.parts[-4:]]]
        for token in candidates:
            match = self.LUX_PATTERN.search(token)
            if match:
                return int(match.group(1))
        return None

    def _group_id_for(self, image_path: Path) -> str:
        cleaned = re.sub(r"(?i)(?:^|[_-])lux[_-]?\d+", "", image_path.stem)
        cleaned = re.sub(r"[_-]{2,}", "_", cleaned).strip("_-")
        return cleaned or image_path.stem

    def _shot_id_for(self, image_path: Path) -> str | None:
        match = self.SHOT_PATTERN.search(image_path.stem)
        if match:
            return match.group(1).replace("-", "_")
        return None

    def _split_for(self, image_path: Path) -> str:
        for part in image_path.parts:
            lowered = part.lower()
            if lowered in {"train", "val", "test"}:
                return lowered
        return "train"


defect_dataset_ingest_service = DefectDatasetIngestService()
