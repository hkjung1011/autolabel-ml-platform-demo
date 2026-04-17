from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.domain.defect_autolabel_models import DefectQualityMetrics

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


class DefectQualityService:
    def load_rgb(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def load_gray(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("L")
        return np.asarray(image, dtype=np.uint8)

    def build_metrics(self, asset_id: str, image_path: str | Path) -> DefectQualityMetrics:
        gray = self.load_gray(image_path).astype(np.float32)
        mean_luma = float(gray.mean())
        median_luma = float(np.median(gray))
        std_luma = float(gray.std())
        shadow_clip_ratio = float(np.mean(gray <= 12.0))
        highlight_clip_ratio = float(np.mean(gray >= 243.0))

        blurred = self._blur_gray(gray.astype(np.uint8)).astype(np.float32)
        local_contrast = np.abs(gray - blurred)
        local_contrast_score = float(np.clip(local_contrast.std() / 32.0, 0.0, 1.0))

        if cv2 is not None:
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)  # type: ignore[union-attr]
            laplacian_var = float(laplacian.var())
        else:  # pragma: no cover - fallback
            gy, gx = np.gradient(gray)
            laplacian_var = float(((gx**2) + (gy**2)).var())
        laplacian_blur_score = float(np.clip(laplacian_var / 250.0, 0.0, 1.0))

        glare_mask = ((gray >= 230.0) & (local_contrast >= 18.0)).astype(np.float32)
        specular_glare_score = float(np.clip((highlight_clip_ratio * 1.5) + (glare_mask.mean() * 2.0), 0.0, 1.0))

        exposure_center_score = 1.0 - min(abs(mean_luma - 128.0) / 128.0, 1.0)
        vision_ready_score = (
            (0.30 * exposure_center_score)
            + (0.25 * local_contrast_score)
            + (0.20 * laplacian_blur_score)
            + (0.15 * (1.0 - shadow_clip_ratio))
            + (0.10 * (1.0 - highlight_clip_ratio))
        )
        lux_bucket = self.classify_lux_bucket(mean_luma=mean_luma, highlight_clip_ratio=highlight_clip_ratio)

        return DefectQualityMetrics(
            asset_id=asset_id,
            mean_luma=round(mean_luma, 4),
            median_luma=round(median_luma, 4),
            std_luma=round(std_luma, 4),
            shadow_clip_ratio=round(shadow_clip_ratio, 6),
            highlight_clip_ratio=round(highlight_clip_ratio, 6),
            local_contrast_score=round(local_contrast_score, 4),
            laplacian_blur_score=round(laplacian_blur_score, 4),
            specular_glare_score=round(specular_glare_score, 4),
            vision_ready_score=round(float(np.clip(vision_ready_score, 0.0, 1.0)), 4),
            lux_bucket=lux_bucket,
        )

    def build_normalized_view(self, gray: np.ndarray) -> np.ndarray:
        gray_uint8 = gray.astype(np.uint8)
        if cv2 is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore[union-attr]
            return clahe.apply(gray_uint8)
        image = Image.fromarray(gray_uint8)
        return np.asarray(ImageOps.autocontrast(image), dtype=np.uint8)

    def build_edge_view(self, gray: np.ndarray) -> np.ndarray:
        gray_uint8 = gray.astype(np.uint8)
        if cv2 is not None:
            blurred = cv2.GaussianBlur(gray_uint8, (0, 0), 3.0)  # type: ignore[union-attr]
            sharpened = cv2.addWeighted(gray_uint8, 1.6, blurred, -0.6, 0.0)  # type: ignore[union-attr]
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        image = Image.fromarray(gray_uint8).filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
        return np.asarray(image, dtype=np.uint8)

    def classify_lux_bucket(self, *, mean_luma: float, highlight_clip_ratio: float) -> str:
        if mean_luma >= 220.0 or highlight_clip_ratio >= 0.08:
            return "glare"
        if mean_luma < 45.0:
            return "very_dark"
        if mean_luma < 80.0:
            return "dark"
        if mean_luma < 170.0:
            return "normal"
        return "bright"

    def _blur_gray(self, gray: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.GaussianBlur(gray, (0, 0), 7.0)  # type: ignore[union-attr]
        return np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.uint8)


defect_quality_service = DefectQualityService()
