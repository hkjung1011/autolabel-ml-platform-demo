from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_json
from app.domain.research_models import FusionVariantResult, PairGroup
from app.plugins.fusion.defect_prior import edge_defect_prior, laplacian, to_gray
from app.plugins.fusion.frequency_split import normalize_signed_image, split_low_high


class DefectAwareFusionProvider:
    name = "defect_aware_fusion"

    def __init__(
        self,
        alpha: float = 0.6,
        high_frequency_sigma: float = 3.0,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 1.0,
        exposedness_sigma: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.high_frequency_sigma = high_frequency_sigma
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
            raise ValueError(f"{group.key}: at least two exposures are required for defect-aware fusion")

        artifact_paths: dict[str, str] = {}
        if output_path.exists() and not overwrite:
            fused_array = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0
            weight_map_paths = self._existing_weight_map_paths(output_dir, valid_luxes)
        else:
            low_stack = []
            high_stack = []
            prior_stack = []
            for image in image_arrays:
                low, high = split_low_high(image, sigma=self.high_frequency_sigma)
                low_stack.append(low)
                high_stack.append(high)
                prior_stack.append(edge_defect_prior(image))

            low_weights = self._compute_mertens_weights(low_stack)
            normalized_low_weights = low_weights / np.clip(np.sum(low_weights, axis=0, keepdims=True), 1e-9, None)
            low_fused = np.sum(normalized_low_weights[..., None] * np.stack(low_stack, axis=0), axis=0)

            high_weights = self._compute_high_frequency_weights(high_stack, prior_stack)
            normalized_high_weights = high_weights / np.clip(np.sum(high_weights, axis=0, keepdims=True), 1e-9, None)
            high_fused = np.sum(normalized_high_weights[..., None] * np.stack(high_stack, axis=0), axis=0)
            fused_array = np.clip(low_fused + high_fused, 0.0, 1.0)

            atomic_save_image(output_path, Image.fromarray(np.clip(fused_array * 255.0, 0, 255).astype(np.uint8)))
            weight_map_paths = self._save_weight_maps(output_dir, normalized_high_weights, valid_luxes)
            if emit_debug_artifacts:
                artifact_paths = self._save_debug_artifacts(
                    output_dir=output_dir,
                    prior_stack=prior_stack,
                    low_fused=low_fused,
                    high_fused=high_fused,
                    source_luxes=valid_luxes,
                )

        gray = to_gray(fused_array)
        result = FusionVariantResult(
            group_id=group.key,
            source_luxes=valid_luxes,
            output_path=str(output_path),
            weight_map_paths=weight_map_paths,
            artifact_paths=artifact_paths,
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
                    "alpha": self.alpha,
                    "high_frequency_sigma": self.high_frequency_sigma,
                    "contrast_weight": self.contrast_weight,
                    "saturation_weight": self.saturation_weight,
                    "exposure_weight": self.exposure_weight,
                    "exposedness_sigma": self.exposedness_sigma,
                },
                "metrics": result.model_dump(),
            },
        )
        return result

    def _compute_mertens_weights(self, image_arrays: list[np.ndarray]) -> np.ndarray:
        weights = []
        for image in image_arrays:
            gray = to_gray(image)
            contrast = np.abs(laplacian(gray)) + 1e-6
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

    def _compute_high_frequency_weights(self, high_stack: list[np.ndarray], prior_stack: list[np.ndarray]) -> np.ndarray:
        weights = []
        for high_image, prior_map in zip(high_stack, prior_stack, strict=True):
            detail = np.abs(laplacian(to_gray(normalize_signed_image(high_image)))) + 1e-6
            weight = detail * (1.0 + (self.alpha * prior_map))
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

    def _save_debug_artifacts(
        self,
        output_dir: Path,
        prior_stack: list[np.ndarray],
        low_fused: np.ndarray,
        high_fused: np.ndarray,
        source_luxes: list[str],
    ) -> dict[str, str]:
        artifact_paths: dict[str, str] = {}
        for prior_map, lux in zip(prior_stack, source_luxes, strict=True):
            prior_path = output_dir / f"prior_lux{lux}.png"
            atomic_save_image(prior_path, Image.fromarray(np.clip(prior_map * 255.0, 0, 255).astype(np.uint8)))
            artifact_paths[f"prior_lux{lux}"] = str(prior_path)

        low_path = output_dir / "low_fused.png"
        high_path = output_dir / "high_fused.png"
        atomic_save_image(low_path, Image.fromarray(np.clip(low_fused * 255.0, 0, 255).astype(np.uint8)))
        atomic_save_image(
            high_path,
            Image.fromarray(np.clip(normalize_signed_image(high_fused) * 255.0, 0, 255).astype(np.uint8)),
        )
        artifact_paths["low_fused"] = str(low_path)
        artifact_paths["high_fused"] = str(high_path)
        return artifact_paths

    def _existing_weight_map_paths(self, output_dir: Path, source_luxes: list[str]) -> dict[str, str]:
        weight_map_paths: dict[str, str] = {}
        for lux in source_luxes:
            path = output_dir / f"weights_lux{lux}.png"
            if path.exists():
                weight_map_paths[lux] = str(path)
        return weight_map_paths
