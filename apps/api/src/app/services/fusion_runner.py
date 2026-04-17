from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from app.domain.research_models import FusionJobResponse, FusionRunRequest, FusionSkippedItem, PairGroup
from app.plugins.fusion.defect_aware import DefectAwareFusionProvider
from app.plugins.fusion.mertens_baseline import MertensFusionProvider


class FusionRunnerService:
    def __init__(self) -> None:
        self.jobs: dict[str, FusionJobResponse] = {}

    def run_mertens(self, request: FusionRunRequest) -> FusionJobResponse:
        provider = MertensFusionProvider(
            contrast_weight=float(request.params.get("contrast_weight", 1.0)),
            saturation_weight=float(request.params.get("saturation_weight", 1.0)),
            exposure_weight=float(request.params.get("exposure_weight", 1.0)),
            exposedness_sigma=float(request.params.get("exposedness_sigma", 0.2)),
        )
        return self._run_with_provider(request, provider)

    def run_daf(self, request: FusionRunRequest) -> FusionJobResponse:
        provider = DefectAwareFusionProvider(
            alpha=float(request.params.get("alpha", 0.6)),
            high_frequency_sigma=float(request.params.get("high_frequency_sigma", 3.0)),
            contrast_weight=float(request.params.get("contrast_weight", 1.0)),
            saturation_weight=float(request.params.get("saturation_weight", 1.0)),
            exposure_weight=float(request.params.get("exposure_weight", 1.0)),
            exposedness_sigma=float(request.params.get("exposedness_sigma", 0.2)),
        )
        return self._run_with_provider(request, provider)

    def _run_with_provider(self, request: FusionRunRequest, provider) -> FusionJobResponse:
        workspace_root = Path(request.workspace_root)
        manifest_name = "labeled_pair_manifest.json" if request.use_labeled_only else "pair_manifest.json"
        manifest_path = workspace_root / "manifests" / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        group_lookup = {group.key: group for group in groups}
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        output_root = workspace_root / "variants" / provider.name
        output_root.mkdir(parents=True, exist_ok=True)
        dataset_root = workspace_root / "datasets" / "yolo_fusion" / provider.name

        outputs = []
        errors = []
        skipped = []
        for group in groups:
            try:
                available = [lux for lux in request.source_luxes if lux in group.exposures]
                if len(available) < 2:
                    skipped.append(
                        FusionSkippedItem(
                            group_id=group.key,
                            reason="insufficient_exposures",
                            available_luxes=sorted(group.exposures.keys()),
                            requested_luxes=list(request.source_luxes),
                            required_min=2,
                        )
                    )
                    continue
                outputs.append(
                    provider.create_variant(
                        workspace_root=workspace_root,
                        group=group,
                        source_luxes=available,
                        overwrite=request.overwrite,
                        emit_debug_artifacts=request.emit_debug_artifacts,
                    )
                )
            except Exception as exc:  # pragma: no cover - returned to API
                errors.append(f"{group.key}: {exc}")

        self._materialize_fusion_dataset(dataset_root=dataset_root, outputs=outputs, group_lookup=group_lookup, variant_name=provider.name)

        job = FusionJobResponse(
            job_id=f"fusion_{uuid4().hex[:12]}",
            status="completed" if not errors else ("partial" if outputs else "failed"),
            method=request.method,
            workspace_root=str(workspace_root),
            total_groups=len(groups),
            processed_groups=len(outputs),
            outputs=outputs,
            skipped=skipped,
            errors=errors,
            output_root=str(output_root),
            dataset_root=str(dataset_root),
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> FusionJobResponse:
        return self.jobs[job_id]

    def _materialize_fusion_dataset(
        self,
        dataset_root: Path,
        outputs: list,
        group_lookup: dict[str, PairGroup],
        variant_name: str,
    ) -> None:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        for output in outputs:
            group = group_lookup.get(output.group_id)
            if group is None or not group.anchor_label_path:
                continue
            split = group.frozen_split or group.split
            stem_source = Path(group.anchor_image_path or output.output_path).stem
            image_destination = dataset_root / "images" / split / f"{stem_source}__{variant_name}.png"
            label_destination = dataset_root / "labels" / split / f"{stem_source}__{variant_name}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output.output_path, image_destination)
            shutil.copy2(group.anchor_label_path, label_destination)


fusion_runner_service = FusionRunnerService()
