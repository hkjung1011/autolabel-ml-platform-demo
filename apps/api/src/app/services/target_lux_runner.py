from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_json
from app.domain.research_models import (
    PairGroup,
    TargetLuxJobResponse,
    TargetLuxRunRequest,
    TargetLuxVariantResult,
)


class TargetLuxRunnerService:
    def __init__(self) -> None:
        self.jobs: dict[str, TargetLuxJobResponse] = {}

    def run(self, request: TargetLuxRunRequest) -> TargetLuxJobResponse:
        workspace_root = Path(request.workspace_root)
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing labeled manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        output_root = workspace_root / "variants" / f"target_lux_{request.target_lux}"
        output_root.mkdir(parents=True, exist_ok=True)

        outputs: list[TargetLuxVariantResult] = []
        errors: list[str] = []
        for group in groups:
            try:
                outputs.append(self._create_variant(workspace_root, group, request))
            except Exception as exc:
                errors.append(f"{group.key}: {exc}")

        job = TargetLuxJobResponse(
            job_id=f"target_lux_{uuid4().hex[:12]}",
            status="completed" if not errors else ("partial" if outputs else "failed"),
            workspace_root=str(workspace_root),
            target_lux=request.target_lux,
            total_groups=len(groups),
            processed_groups=len(outputs),
            outputs=outputs,
            errors=errors,
            output_root=str(output_root),
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> TargetLuxJobResponse:
        return self.jobs[job_id]

    def _create_variant(self, workspace_root: Path, group: PairGroup, request: TargetLuxRunRequest) -> TargetLuxVariantResult:
        safe_group_id = group.key.replace("|", "__")
        output_dir = workspace_root / "variants" / f"target_lux_{request.target_lux}" / safe_group_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fused.png"
        metadata_path = output_dir / "metadata.json"

        available = []
        for lux in request.source_luxes:
            source_path = group.exposures.get(lux)
            if source_path and Path(source_path).exists():
                available.append((int(lux), Path(source_path)))
        if len(available) < 1:
            raise ValueError("No valid source lux exposures available.")

        if output_path.exists() and not request.overwrite:
            image = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            weights = metadata.get("blend_weights", {})
            source_used = metadata.get("source_luxes_used", [])
        else:
            image, weights, source_used = self._blend_to_target(available, request.target_lux)
            if request.apply_clahe:
                image = self._apply_clahe(image, clip_limit=request.clahe_clip_limit, grid_size=request.clahe_grid_size)
            atomic_save_image(output_path, Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)))

        gray = self._to_gray(image)
        result = TargetLuxVariantResult(
            group_id=group.key,
            target_lux=request.target_lux,
            source_luxes_used=source_used,
            blend_weights=weights,
            output_path=str(output_path),
            mean_intensity=round(float(image.mean()), 4),
            dynamic_range_score=round(float(np.percentile(gray, 95) - np.percentile(gray, 5)), 4),
            output_size_bytes=output_path.stat().st_size,
        )
        atomic_write_json(
            metadata_path,
            {
                "group_id": group.key,
                "target_lux": request.target_lux,
                "apply_clahe": request.apply_clahe,
                "clahe_clip_limit": request.clahe_clip_limit,
                "clahe_grid_size": request.clahe_grid_size,
                "source_luxes_used": source_used,
                "blend_weights": weights,
                "metrics": result.model_dump(),
            },
        )
        return result

    def _blend_to_target(self, available: list[tuple[int, Path]], target_lux: int) -> tuple[np.ndarray, dict[str, float], list[str]]:
        available = sorted(available, key=lambda item: item[0])
        lux_values = [item[0] for item in available]

        if target_lux <= lux_values[0]:
            image = self._load_image(available[0][1])
            scale = max(0.5, target_lux / max(lux_values[0], 1))
            return np.clip(image * scale, 0.0, 1.0), {str(lux_values[0]): 1.0}, [str(lux_values[0])]
        if target_lux >= lux_values[-1]:
            image = self._load_image(available[-1][1])
            scale = min(1.5, target_lux / max(lux_values[-1], 1))
            return np.clip(image * scale, 0.0, 1.0), {str(lux_values[-1]): 1.0}, [str(lux_values[-1])]

        lower = available[0]
        upper = available[-1]
        for current, nxt in zip(available, available[1:], strict=False):
            if current[0] <= target_lux <= nxt[0]:
                lower, upper = current, nxt
                break

        lower_log = np.log(max(lower[0], 1))
        upper_log = np.log(max(upper[0], 1))
        target_log = np.log(max(target_lux, 1))
        upper_weight = (target_log - lower_log) / max(upper_log - lower_log, 1e-6)
        lower_weight = 1.0 - upper_weight

        lower_image = self._load_image(lower[1])
        upper_image = self._load_image(upper[1])
        blended = np.clip((lower_image * lower_weight) + (upper_image * upper_weight), 0.0, 1.0)
        return blended, {
            str(lower[0]): round(float(lower_weight), 4),
            str(upper[0]): round(float(upper_weight), 4),
        }, [str(lower[0]), str(upper[0])]

    def _apply_clahe(self, image: np.ndarray, *, clip_limit: float, grid_size: int) -> np.ndarray:
        bgr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return np.asarray(rgb, dtype=np.float32) / 255.0

    def _load_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])


target_lux_runner_service = TargetLuxRunnerService()
