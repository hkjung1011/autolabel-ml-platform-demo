from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_json
from app.domain.research_models import (
    ForensicWdrJobResponse,
    ForensicWdrRunRequest,
    ForensicWdrVariantResult,
    PairGroup,
)


class ForensicWdrService:
    def __init__(self) -> None:
        self.jobs: dict[str, ForensicWdrJobResponse] = {}

    def run(self, request: ForensicWdrRunRequest) -> ForensicWdrJobResponse:
        workspace_root = Path(request.workspace_root)
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing labeled manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        output_root = workspace_root / "variants" / "forensic_wdr"
        output_root.mkdir(parents=True, exist_ok=True)

        outputs: list[ForensicWdrVariantResult] = []
        errors: list[str] = []
        for group in groups:
            try:
                outputs.append(self._create_variant(workspace_root, group, request))
            except Exception as exc:
                errors.append(f"{group.key}: {exc}")

        job = ForensicWdrJobResponse(
            job_id=f"forensic_wdr_{uuid4().hex[:12]}",
            status="completed" if not errors else ("partial" if outputs else "failed"),
            workspace_root=str(workspace_root),
            total_groups=len(groups),
            processed_groups=len(outputs),
            outputs=outputs,
            errors=errors,
            output_root=str(output_root),
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ForensicWdrJobResponse:
        return self.jobs[job_id]

    def _create_variant(self, workspace_root: Path, group: PairGroup, request: ForensicWdrRunRequest) -> ForensicWdrVariantResult:
        safe_group_id = group.key.replace("|", "__")
        output_dir = workspace_root / "variants" / "forensic_wdr" / safe_group_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fused.png"
        metadata_path = output_dir / "metadata.json"

        available: list[tuple[int, Path]] = []
        for lux in request.source_luxes:
            source_path = group.exposures.get(lux)
            if source_path and Path(source_path).exists():
                available.append((int(lux), Path(source_path)))
        if len(available) < 2:
            raise ValueError("Forensic WDR requires at least two valid lux exposures.")

        if output_path.exists() and not request.overwrite:
            fused = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            source_used = metadata.get("source_luxes_used", [])
        else:
            fused = self._build_forensic_wdr(available, request)
            atomic_save_image(output_path, Image.fromarray(np.clip(fused * 255.0, 0, 255).astype(np.uint8)))
            source_used = [str(lux) for lux, _ in available]

        gray = self._to_gray(fused)
        result = ForensicWdrVariantResult(
            group_id=group.key,
            source_luxes_used=source_used,
            output_path=str(output_path),
            mean_intensity=round(float(fused.mean()), 4),
            dynamic_range_score=round(float(np.percentile(gray, 95) - np.percentile(gray, 5)), 4),
            local_contrast_score=round(float(cv2.Laplacian((gray * 255.0).astype(np.uint8), cv2.CV_32F).var() / 255.0), 4),
            output_size_bytes=output_path.stat().st_size,
        )
        atomic_write_json(
            metadata_path,
            {
                "group_id": group.key,
                "source_luxes_used": source_used,
                "detail_boost": request.detail_boost,
                "shadow_lift": request.shadow_lift,
                "highlight_protect": request.highlight_protect,
                "clahe_clip_limit": request.clahe_clip_limit,
                "bilateral_sigma": request.bilateral_sigma,
                "metrics": result.model_dump(),
            },
        )
        return result

    def _build_forensic_wdr(self, available: list[tuple[int, Path]], request: ForensicWdrRunRequest) -> np.ndarray:
        ordered = sorted(available, key=lambda item: item[0])
        min_lux = ordered[0][0]
        max_lux = ordered[-1][0]
        images = [self._load_image(path) for _, path in ordered]
        grays = [self._to_gray(image) for image in images]
        detail_maps = [self._normalize(np.abs(cv2.Laplacian((gray * 255.0).astype(np.uint8), cv2.CV_32F))) for gray in grays]

        weights = []
        for (lux, _), gray, detail in zip(ordered, grays, detail_maps, strict=False):
            if max_lux == min_lux:
                exposure_rank = 0.5
            else:
                exposure_rank = (lux - min_lux) / max(max_lux - min_lux, 1)

            exposedness = np.exp(-((gray - 0.5) ** 2) / (2 * (0.2**2)))
            highlight_mask = np.clip((gray - 0.62) / 0.28, 0.0, 1.0)
            shadow_mask = np.clip((0.38 - gray) / 0.38, 0.0, 1.0)
            highlight_pref = (1.0 - exposure_rank) * request.highlight_protect * highlight_mask
            shadow_pref = exposure_rank * request.shadow_lift * shadow_mask
            weight = 1e-3 + exposedness + (detail * request.detail_boost) + highlight_pref + shadow_pref
            weights.append(weight)

        weight_stack = np.stack(weights, axis=2)
        weight_stack /= np.clip(weight_stack.sum(axis=2, keepdims=True), 1e-6, None)
        fused = np.zeros_like(images[0], dtype=np.float32)
        for image, weight in zip(images, np.moveaxis(weight_stack, 2, 0), strict=False):
            fused += image * weight[:, :, None]

        fused = self._forensic_tone_balance(
            fused,
            clip_limit=request.clahe_clip_limit,
            bilateral_sigma=request.bilateral_sigma,
        )
        return np.clip(fused, 0.0, 1.0)

    def _forensic_tone_balance(self, image: np.ndarray, *, clip_limit: float, bilateral_sigma: float) -> np.ndarray:
        bgr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        l_channel = cv2.bilateralFilter(l_channel, d=0, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return np.asarray(rgb, dtype=np.float32) / 255.0

    def _load_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value - min_value < 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_value) / (max_value - min_value)).astype(np.float32)


forensic_wdr_service = ForensicWdrService()
