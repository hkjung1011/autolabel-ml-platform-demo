from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_json
from app.domain.research_models import PairGroup, RetinexVariantResult
from app.plugins.retinex.config import DEFAULT_MSRCR_CONFIG, MSRCRConfig
from app.plugins.retinex.msrcr import apply_msrcr


class RetinexEnhancementProvider:
    name = "retinex_msrcr"

    def __init__(self, config: MSRCRConfig | None = None) -> None:
        self.config = config or DEFAULT_MSRCR_CONFIG

    def create_variant(
        self,
        workspace_root: Path,
        group: PairGroup,
        source_lux: str,
        overwrite: bool = True,
    ) -> RetinexVariantResult:
        source_path = Path(group.exposures[source_lux])
        output_dir = workspace_root / "variants" / "retinex_msrcr" / self._safe_group_dir(group.key)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"lux{source_lux}_restored.png"
        metadata_path = output_dir / f"lux{source_lux}_metadata.json"

        if output_path.exists() and not overwrite:
            restored_array = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32)
        else:
            source_image = Image.open(source_path).convert("RGB")
            restored_array = apply_msrcr(source_image, self.config).astype(np.float32)
            anchor_path = Path(group.anchor_image_path) if group.anchor_image_path else None
            if anchor_path is not None and anchor_path.exists():
                restored_array = self._match_anchor_brightness(restored_array, anchor_path)
            atomic_save_image(output_path, Image.fromarray(restored_array.astype(np.uint8)))

        anchor_path = Path(group.anchor_image_path) if group.anchor_image_path else None
        brightness_delta = 0.0
        ssim = None
        if anchor_path is not None and anchor_path.exists():
            anchor_array = np.asarray(Image.open(anchor_path).convert("RGB"), dtype=np.float32)
            restored_mean = float(restored_array.mean() / 255.0)
            anchor_mean = float(anchor_array.mean() / 255.0)
            brightness_delta = round(restored_mean - anchor_mean, 4)
            ssim = round(self._ssim_gray(restored_array, anchor_array), 4)
        mean_intensity = round(float(restored_array.mean() / 255.0), 4)

        result = RetinexVariantResult(
            group_id=group.key,
            source_lux=source_lux,
            output_path=str(output_path),
            anchor_path=str(anchor_path) if anchor_path is not None else None,
            accepted=(ssim is None or ssim >= 0.45),
            brightness_delta=brightness_delta,
            ssim_vs_anchor=ssim,
            mean_intensity=mean_intensity,
            output_size_bytes=output_path.stat().st_size,
        )
        atomic_write_json(
            metadata_path,
            {
                "provider": self.name,
                "group_id": group.key,
                "source_lux": source_lux,
                "config": {
                    "sigma_list": self.config.sigma_list,
                    "gain": self.config.gain,
                    "offset": self.config.offset,
                    "alpha": self.config.alpha,
                    "beta": self.config.beta,
                    "low_clip_percentile": self.config.low_clip_percentile,
                    "high_clip_percentile": self.config.high_clip_percentile,
                },
                "metrics": result.model_dump(),
            },
        )
        return result

    def _match_anchor_brightness(self, restored_array: np.ndarray, anchor_path: Path) -> np.ndarray:
        anchor_array = np.asarray(Image.open(anchor_path).convert("RGB"), dtype=np.float32)
        target_mean = float(anchor_array.mean())
        current_mean = float(restored_array.mean()) or 1.0
        scale = max(0.6, min(1.8, target_mean / current_mean))
        return np.clip(restored_array * scale, 0, 255)

    def _ssim_gray(self, left: np.ndarray, right: np.ndarray) -> float:
        left_gray = self._to_gray(left)
        right_gray = self._to_gray(right)
        if left_gray.shape != right_gray.shape:
            min_h = min(left_gray.shape[0], right_gray.shape[0])
            min_w = min(left_gray.shape[1], right_gray.shape[1])
            left_gray = left_gray[:min_h, :min_w]
            right_gray = right_gray[:min_h, :min_w]

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu_x = float(left_gray.mean())
        mu_y = float(right_gray.mean())
        sigma_x = float(left_gray.var())
        sigma_y = float(right_gray.var())
        sigma_xy = float(((left_gray - mu_x) * (right_gray - mu_y)).mean())
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        return numerator / denominator if denominator else 0.0

    def _to_gray(self, array: np.ndarray) -> np.ndarray:
        return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]

    def _safe_group_dir(self, group_id: str) -> str:
        return group_id.replace("|", "__")
