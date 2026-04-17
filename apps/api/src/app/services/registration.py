from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    PairGroup,
    RegistrationJobResponse,
    RegistrationReport,
    RegistrationVerifyRequest,
)
from app.plugins.registration.translation_aligner import estimate_translation, load_grayscale_array
from app.plugins.registration.verifier import label_iou_drift, read_yolo_boxes


class RegistrationService:
    def __init__(self) -> None:
        self.jobs: dict[str, RegistrationJobResponse] = {}

    def run(self, request: RegistrationVerifyRequest) -> RegistrationJobResponse:
        workspace_root = Path(request.workspace_root)
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing labeled manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        reports: list[RegistrationReport] = []
        errors: list[str] = []
        group_lookup = {group.key: group for group in groups}

        for group in groups:
            if not group.anchor_image_path or not group.anchor_label_path:
                errors.append(f"{group.key}: missing anchor image or label")
                continue
            for source_lux in request.source_luxes:
                source_path_str = group.exposures.get(source_lux)
                if source_path_str is None:
                    errors.append(f"{group.key}: missing raw lux{source_lux} exposure")
                    continue
                source_path = Path(source_path_str)
                try:
                    variant_path = self._resolve_variant_path(workspace_root, request.variant_source, group, source_lux)
                except FileNotFoundError as exc:
                    errors.append(str(exc))
                    continue

                try:
                    reports.append(
                        self._verify_one(
                            group=group,
                            variant_source=request.variant_source,
                            source_path=source_path,
                            variant_path=variant_path,
                            source_lux=source_lux,
                            max_shift_px=request.max_shift_px,
                            max_corner_error_px=request.max_corner_error_px,
                            max_label_iou_drift=request.max_label_iou_drift,
                            max_variant_shift_px=request.max_variant_shift_px,
                            max_variant_iou_drift=request.max_variant_iou_drift,
                            min_similarity=request.min_similarity,
                        )
                    )
                except Exception as exc:  # pragma: no cover - surfaced via API
                    errors.append(f"{group.key}/lux{source_lux}: {exc}")

        accepted_count = sum(1 for report in reports if report.status == "accept")
        warning_count = sum(1 for report in reports if report.status == "warning")
        rejected_count = sum(1 for report in reports if report.status == "reject")

        report_path = workspace_root / "registration_reports" / f"{request.variant_source}.json"
        accepted_manifest_path = workspace_root / "registration_reports" / f"{request.variant_source}_accepted_manifest.json"
        registered_dataset_root = workspace_root / "datasets" / "registered_variants" / request.variant_source
        accepted_keys = [report.group_id for report in reports if report.status == "accept"]
        warning_keys = [report.group_id for report in reports if report.status == "warning"]
        rejected_keys = [report.group_id for report in reports if report.status == "reject"]
        report_payload = {
            "variant_source": request.variant_source,
            "workspace_root": str(workspace_root),
            "total_groups": len(groups),
            "processed_groups": len(reports),
            "accepted_count": accepted_count,
            "warning_count": warning_count,
            "rejected_count": rejected_count,
            "accepted_keys": accepted_keys,
            "warning_keys": warning_keys,
            "rejected_keys": rejected_keys,
            "errors": errors,
            "reports": [report.model_dump() for report in reports],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(report_path, json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        atomic_write_text(
            accepted_manifest_path,
            json.dumps(
                {
                    "variant_source": request.variant_source,
                    "workspace_root": str(workspace_root),
                    "accepted_keys": accepted_keys,
                    "warning_keys": warning_keys,
                    "rejected_keys": rejected_keys,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if request.materialize_accepted_dataset:
            self._materialize_registered_dataset(
                dataset_root=registered_dataset_root,
                reports=reports,
                group_lookup=group_lookup,
                variant_source=request.variant_source,
            )

        job = RegistrationJobResponse(
            job_id=f"registration_{uuid4().hex[:12]}",
            status="completed" if not errors else ("partial" if reports else "failed"),
            variant_source=request.variant_source,
            workspace_root=str(workspace_root),
            total_groups=len(groups),
            processed_groups=len(reports),
            accepted_count=accepted_count,
            warning_count=warning_count,
            rejected_count=rejected_count,
            reports=reports,
            errors=errors,
            report_path=str(report_path),
            accepted_manifest_path=str(accepted_manifest_path),
            registered_dataset_root=str(registered_dataset_root) if request.materialize_accepted_dataset else None,
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> RegistrationJobResponse:
        return self.jobs[job_id]

    def load_report(self, workspace_root: str, variant_source: str) -> dict:
        report_path = Path(workspace_root) / "registration_reports" / f"{variant_source}.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing registration report: {report_path}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    def _resolve_variant_path(self, workspace_root: Path, variant_source: str, group: PairGroup, source_lux: str) -> Path:
        safe_group_id = group.key.replace("|", "__")
        if variant_source == "retinex_msrcr":
            variant_path = workspace_root / "variants" / "retinex_msrcr" / safe_group_id / f"lux{source_lux}_restored.png"
        elif variant_source == "raw":
            source_path = group.exposures.get(source_lux)
            if source_path is None:
                raise FileNotFoundError(f"{group.key}: missing raw lux{source_lux} exposure")
            variant_path = Path(source_path)
        else:
            raise ValueError(f"Unsupported variant source: {variant_source}")

        if not variant_path.exists():
            raise FileNotFoundError(f"{group.key}: missing variant output {variant_path}")
        return variant_path

    def _verify_one(
        self,
        group: PairGroup,
        variant_source: str,
        source_path: Path,
        variant_path: Path,
        source_lux: str,
        max_shift_px: int,
        max_corner_error_px: float,
        max_label_iou_drift: float,
        max_variant_shift_px: float,
        max_variant_iou_drift: float,
        min_similarity: float,
    ) -> RegistrationReport:
        anchor_path = Path(group.anchor_image_path or "")
        reference = load_grayscale_array(str(anchor_path))
        source_array = load_grayscale_array(str(source_path))
        variant_array = load_grayscale_array(str(variant_path))
        source_dx_px, source_dy_px, source_similarity = estimate_translation(
            reference=reference,
            moving=source_array,
            max_shift_px=max_shift_px,
        )
        variant_dx_px, variant_dy_px, variant_similarity = estimate_translation(
            reference=source_array,
            moving=variant_array,
            max_shift_px=max_shift_px,
        )
        dx_px = source_dx_px + variant_dx_px
        dy_px = source_dy_px + variant_dy_px
        similarity = min(source_similarity, variant_similarity)
        mean_corner_error_px = (dx_px**2 + dy_px**2) ** 0.5
        variant_only_corner_error_px = (variant_dx_px**2 + variant_dy_px**2) ** 0.5

        with Image.open(anchor_path) as image:
            image_width, image_height = image.size
        boxes = read_yolo_boxes(Path(group.anchor_label_path or ""))
        iou_drift = label_iou_drift(boxes, image_width=image_width, image_height=image_height, dx_px=dx_px, dy_px=dy_px)
        variant_iou_drift = label_iou_drift(
            boxes,
            image_width=image_width,
            image_height=image_height,
            dx_px=variant_dx_px,
            dy_px=variant_dy_px,
        )
        status = self._classify_status(
            source_similarity=source_similarity,
            variant_similarity=variant_similarity,
            mean_corner_error_px=mean_corner_error_px,
            label_iou_drift=iou_drift,
            variant_only_corner_error_px=variant_only_corner_error_px,
            variant_only_iou_drift=variant_iou_drift,
            max_corner_error_px=max_corner_error_px,
            max_label_iou_drift=max_label_iou_drift,
            max_variant_shift_px=max_variant_shift_px,
            max_variant_iou_drift=max_variant_iou_drift,
            min_similarity=min_similarity,
        )

        return RegistrationReport(
            group_id=group.key,
            source_lux=source_lux,
            variant_source=variant_source,
            source_path=str(source_path),
            variant_path=str(variant_path),
            anchor_path=str(anchor_path),
            status=status,
            similarity=round(similarity, 4),
            dx_px=dx_px,
            dy_px=dy_px,
            mean_corner_error_px=round(mean_corner_error_px, 4),
            label_iou_drift=round(iou_drift, 4),
            source_to_anchor_similarity=round(source_similarity, 4),
            source_to_anchor_dx_px=source_dx_px,
            source_to_anchor_dy_px=source_dy_px,
            variant_to_source_similarity=round(variant_similarity, 4),
            variant_to_source_dx_px=variant_dx_px,
            variant_to_source_dy_px=variant_dy_px,
            variant_only_corner_error_px=round(variant_only_corner_error_px, 4),
            variant_only_iou_drift=round(variant_iou_drift, 4),
            warp_matrix=[
                [1.0, 0.0, float(dx_px)],
                [0.0, 1.0, float(dy_px)],
                [0.0, 0.0, 1.0],
            ],
        )

    def _classify_status(
        self,
        source_similarity: float,
        variant_similarity: float,
        mean_corner_error_px: float,
        label_iou_drift: float,
        variant_only_corner_error_px: float,
        variant_only_iou_drift: float,
        max_corner_error_px: float,
        max_label_iou_drift: float,
        max_variant_shift_px: float,
        max_variant_iou_drift: float,
        min_similarity: float,
    ) -> str:
        if (
            variant_similarity >= min_similarity
            and variant_only_corner_error_px <= max_variant_shift_px
            and variant_only_iou_drift <= max_variant_iou_drift
            and source_similarity >= min_similarity
            and mean_corner_error_px <= max_corner_error_px
            and label_iou_drift <= max_label_iou_drift
        ):
            return "accept"
        if (
            variant_similarity >= (min_similarity - 0.05)
            and variant_only_corner_error_px <= (max_variant_shift_px * 1.5)
            and variant_only_iou_drift <= (max_variant_iou_drift * 1.5)
        ):
            return "warning"
        return "reject"

    def _materialize_registered_dataset(
        self,
        dataset_root: Path,
        reports: list[RegistrationReport],
        group_lookup: dict[str, PairGroup],
        variant_source: str,
    ) -> None:
        accepted_reports = [report for report in reports if report.status == "accept"]
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        for report in accepted_reports:
            group = group_lookup.get(report.group_id)
            if group is None or not group.anchor_label_path:
                continue
            split = group.frozen_split or group.split
            source_stem = Path(report.source_path).stem
            image_destination = dataset_root / "images" / split / f"{source_stem}__{variant_source}.png"
            label_destination = dataset_root / "labels" / split / f"{source_stem}__{variant_source}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report.variant_path, image_destination)
            shutil.copy2(group.anchor_label_path, label_destination)


registration_service = RegistrationService()
