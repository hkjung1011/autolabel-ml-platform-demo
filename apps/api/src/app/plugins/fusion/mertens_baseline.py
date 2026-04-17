from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_json
from app.domain.research_models import FusionVariantResult, PairGroup


class MertensFusionProvider:
    name = "mertens_baseline"

    def __init__(
        self,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 1.0,
        exposedness_sigma: float = 0.2,
    ) -> None:
        self.contrast_weight = contrast_weight
        self.saturation_weight = saturation_weight
        self.exposure_weight = exposure_weight
        self.exposedness_sigma = exposedness_sigma

    def create_variant(
        self,
        workspace_root: Path,
        group: PairGroup,
        source_luxes: list[str],
        overwrite: bool = True,
        emit_debug_artifacts: bool = True,
    ) -> FusionVariantResult:
        safe_group_id = group.key.replace("|", "__")
        output_dir = workspace_root / "variants" / self.name / safe_group_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fused.png"
        metadata_path = output_dir / "metadata.json"

        image_arrays = []
        valid_luxes = []
        for lux in source_luxes:
            source_path = group.exposures.get(lux)
            if source_path is None:
                continue
            image = Image.open(source_path).convert("RGB")
            image_arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
            valid_luxes.append(lux)

        if len(image_arrays) < 2:
            raise ValueError(f"{group.key}: at least two exposures are required for Mertens fusion")

        if output_path.exists() and not overwrite:
            fused_array = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0
            weight_map_paths = self._existing_weight_map_paths(output_dir, valid_luxes)
        else:
            weight_stack = self._compute_weight_stack(image_arrays)
            normalized_weights = weight_stack / np.clip(np.sum(weight_stack, axis=0, keepdims=True), 1e-9, None)
            fused_array = np.sum(normalized_weights[..., None] * np.stack(image_arrays, axis=0), axis=0)
            atomic_save_image(output_path, Image.fromarray(np.clip(fused_array * 255.0, 0, 255).astype(np.uint8)))
            weight_map_paths = self._save_weight_maps(output_dir, normalized_weights, valid_luxes)

        gray = self._to_gray(fused_array)
        result = FusionVariantResult(
            group_id=group.key,
            source_luxes=valid_luxes,
            output_path=str(output_path),
            weight_map_paths=weight_map_paths,
            mean_intensity=round(float(fused_array.mean()), 4),
            dynamic_range_score=round(float(np.percentile(gray, 95) - np.percentile(gray, 5)), 4),
            output_size_bytes=output_path.stat().st_size,
        )
        atomic_write_json(
            metadata_path,
            {
                "provider": self.name,
                "group_id": group.key,
                "source_luxes": valid_luxes,
                "params": {
                    "contrast_weight": self.contrast_weight,
                    "saturation_weight": self.saturation_weight,
                    "exposure_weight": self.exposure_weight,
                    "exposedness_sigma": self.exposedness_sigma,
                },
                "metrics": result.model_dump(),
            },
        )
        return result

    def _compute_weight_stack(self, image_arrays: list[np.ndarray]) -> np.ndarray:
        weights = []
        for image in image_arrays:
            gray = self._to_gray(image)
            contrast = np.abs(self._laplacian(gray)) + 1e-6
            saturation = np.std(image, axis=2) + 1e-6
            exposedness = np.ones_like(gray, dtype=np.float32)
            for channel in range(image.shape[2]):
                exposedness *= np.exp(-((image[:, :, channel] - 0.5) ** 2) / (2 * (self.exposedness_sigma**2)))
            weight = (
                (contrast ** self.contrast_weight)
                * (saturation ** self.saturation_weight)
                * (exposedness ** self.exposure_weight)
            )
            weights.append(weight.astype(np.float32))
        return np.stack(weights, axis=0)

    def _save_weight_maps(self, output_dir: Path, normalized_weights: np.ndarray, source_luxes: list[str]) -> dict[str, str]:
        weight_map_paths: dict[str, str] = {}
        for index, lux in enumerate(source_luxes):
            weight_map_path = output_dir / f"weights_lux{lux}.png"
            weight_map = np.clip(normalized_weights[index] * 255.0, 0, 255).astype(np.uint8)
            atomic_save_image(weight_map_path, Image.fromarray(weight_map))
            weight_map_paths[lux] = str(weight_map_path)
        return weight_map_paths

    def _existing_weight_map_paths(self, output_dir: Path, source_luxes: list[str]) -> dict[str, str]:
        weight_map_paths: dict[str, str] = {}
        for lux in source_luxes:
            path = output_dir / f"weights_lux{lux}.png"
            if path.exists():
                weight_map_paths[lux] = str(path)
        return weight_map_paths

    def _laplacian(self, gray: np.ndarray) -> np.ndarray:
        up = np.roll(gray, 1, axis=0)
        down = np.roll(gray, -1, axis=0)
        left = np.roll(gray, 1, axis=1)
        right = np.roll(gray, -1, axis=1)
        return (up + down + left + right) - (4.0 * gray)

    def _to_gray(self, array: np.ndarray) -> np.ndarray:
        return (0.299 * array[:, :, 0]) + (0.587 * array[:, :, 1]) + (0.114 * array[:, :, 2])
