from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.config import settings
from app.domain.models import (
    AnnotationCandidate,
    AnnotationPrediction,
    ArtifactMetric,
    Asset,
    AssetKind,
    AssetVariant,
    BoundingBox,
    ExposureLightingDecision,
    QualityMetric,
    SceneCondition,
    SyntheticRecipe,
    TransformSafety,
    VariantType,
)


class HeuristicSceneAnalyzer:
    name = "heuristic_scene_analyzer"

    def analyze(self, image_path: Path, kind: AssetKind) -> tuple[SceneCondition, QualityMetric]:
        image = Image.open(image_path).convert("RGB")
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        rgb = np.asarray(image, dtype=np.float32)

        brightness_mean = float(gray.mean() / 255.0)
        contrast_std = float(gray.std() / 255.0)
        gradients = np.diff(gray, axis=0, prepend=gray[:1, :]) + np.diff(gray, axis=1, prepend=gray[:, :1])
        sharpness_score = float(np.var(gradients) / 255.0)
        entropy_hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
        entropy_score = float(-(entropy_hist * np.log2(entropy_hist + 1e-9)).sum() / 6.0)

        low_light_score = max(0.0, min(1.0, 1.0 - brightness_mean * 1.6))
        bright_pixels = rgb.max(axis=2)
        dark_pixels = rgb.min(axis=2)
        highlight_clipping_score = float((bright_pixels > 245).mean())
        shadow_crushing_score = float((dark_pixels < 15).mean())
        backlight_score = max(0.0, min(1.0, highlight_clipping_score * 1.5 + shadow_crushing_score * 0.8))
        blur_score = max(0.0, min(1.0, 1.0 - min(1.0, sharpness_score * 2.5)))

        denoised = image.filter(ImageFilter.MedianFilter(size=3))
        residual = np.abs(
            np.asarray(image.convert("L"), dtype=np.float32)
            - np.asarray(denoised.convert("L"), dtype=np.float32)
        )
        noise_score = max(0.0, min(1.0, float(residual.mean() / 40.0)))

        recommended_chain: list[str] = []
        if kind in {AssetKind.RAW, AssetKind.MULTI_EXPOSURE}:
            recommended_chain.append("true_wdr_merge")
        elif backlight_score > 0.35:
            recommended_chain.append("software_wdr")
        if low_light_score > 0.45:
            recommended_chain.append("low_light_enhancement")
        if noise_score > 0.30:
            recommended_chain.append("denoise")
        if blur_score > 0.40:
            recommended_chain.append("deblur")
        if contrast_std < 0.15:
            recommended_chain.append("local_contrast")

        summary = (
            f"Low-light {low_light_score:.2f}, backlight {backlight_score:.2f}, "
            f"blur {blur_score:.2f}, noise {noise_score:.2f}"
        )
        scene = SceneCondition(
            low_light_score=low_light_score,
            backlight_score=backlight_score,
            blur_score=blur_score,
            noise_score=noise_score,
            highlight_clipping_score=highlight_clipping_score,
            shadow_crushing_score=shadow_crushing_score,
            recommended_chain=recommended_chain or ["preserve_original"],
            summary=summary,
        )
        quality = QualityMetric(
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
            sharpness_score=sharpness_score,
            entropy_score=entropy_score,
        )
        return scene, quality


class HeuristicEnhancementProvider:
    name = "heuristic_enhancer"

    def create_variant(
        self,
        asset: Asset,
        image_path: Path,
        scene_condition: SceneCondition,
    ) -> tuple[AssetVariant, QualityMetric, ArtifactMetric]:
        image = Image.open(image_path).convert("RGB")
        chain = scene_condition.recommended_chain
        transformed = image.copy()

        if "software_wdr" in chain or "true_wdr_merge" in chain:
            transformed = ImageOps.autocontrast(transformed, cutoff=1)
            transformed = ImageEnhance.Contrast(transformed).enhance(1.15)
        if "low_light_enhancement" in chain:
            transformed = ImageEnhance.Brightness(transformed).enhance(1.35)
            transformed = ImageEnhance.Color(transformed).enhance(1.05)
        if "denoise" in chain:
            transformed = transformed.filter(ImageFilter.MedianFilter(size=3))
        if "deblur" in chain:
            transformed = transformed.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
        if "local_contrast" in chain:
            transformed = ImageEnhance.Contrast(transformed).enhance(1.12)

        variant_path = settings.variant_dir / f"{asset.id}_enhanced.png"
        transformed.save(variant_path)

        analyzer = HeuristicSceneAnalyzer()
        _, quality = analyzer.analyze(variant_path, asset.kind)
        artifact = ArtifactMetric(
            halo_risk=min(1.0, 0.15 + 0.10 * ("deblur" in chain)),
            color_shift_risk=min(1.0, 0.10 + 0.15 * ("low_light_enhancement" in chain)),
            oversharpen_risk=min(1.0, 0.05 + 0.25 * ("deblur" in chain)),
            noise_amplification_risk=min(1.0, 0.05 + 0.20 * ("software_wdr" in chain and "denoise" not in chain)),
        )
        safety = (
            TransformSafety.REVIEW_REQUIRED
            if {"deblur", "super_resolution"} & set(chain)
            else TransformSafety.LABEL_SAFE
        )
        variant = AssetVariant(
            asset_id=asset.id,
            variant_type=VariantType.ENHANCED,
            name="enhanced",
            file_path=str(variant_path),
            transforms=chain,
            safety=safety,
            quality_metrics=quality,
            artifact_metrics=artifact,
        )
        return variant, quality, artifact


class HeuristicExposureLightingController:
    name = "heuristic_aelc_vision_controller"

    def plan(
        self,
        scene_condition: SceneCondition,
        quality_metrics: QualityMetric,
    ) -> ExposureLightingDecision:
        target_exposure_bias = round(
            max(-1.5, min(1.5, scene_condition.low_light_score * 1.4 - scene_condition.highlight_clipping_score * 0.9)),
            2,
        )
        recommended_gain = round(
            max(1.0, min(8.0, 1.0 + scene_condition.low_light_score * 4.5 - scene_condition.noise_score * 1.5)),
            2,
        )
        recommended_shutter_ratio = round(
            max(0.2, min(1.0, 1.0 - scene_condition.blur_score * 0.45 + scene_condition.low_light_score * 0.15)),
            2,
        )

        if scene_condition.backlight_score > 0.45:
            lighting_action = "enable_backlight_fill"
        elif scene_condition.low_light_score > 0.55:
            lighting_action = "increase_led_fill"
        elif scene_condition.highlight_clipping_score > 0.18:
            lighting_action = "reduce_front_light"
        else:
            lighting_action = "maintain_current_lighting"

        vision_ready_score = round(
            max(
                0.1,
                min(
                    0.99,
                    0.55
                    + quality_metrics.contrast_std * 0.2
                    + quality_metrics.sharpness_score * 0.15
                    - scene_condition.noise_score * 0.18
                    - scene_condition.blur_score * 0.16,
                ),
            ),
            3,
        )
        rationale = (
            f"Exposure bias {target_exposure_bias}, gain {recommended_gain}, "
            f"lighting action {lighting_action} to improve vision-ready score."
        )
        return ExposureLightingDecision(
            control_mode="auto_exposure_lighting_vision_loop",
            target_exposure_bias=target_exposure_bias,
            recommended_gain=recommended_gain,
            recommended_shutter_ratio=recommended_shutter_ratio,
            lighting_action=lighting_action,
            vision_ready_score=vision_ready_score,
            rationale=rationale,
        )


class HeuristicAutoLabelProvider:
    name = "heuristic_labeler"

    def infer(self, variant: AssetVariant, quality_metrics: QualityMetric) -> AnnotationCandidate:
        seed = abs(hash(variant.id)) % 100000
        rng = random.Random(seed)
        base_conf = 0.35 + quality_metrics.brightness_mean * 0.25 + quality_metrics.contrast_std * 0.25
        base_conf += min(0.20, quality_metrics.sharpness_score * 0.10)
        if variant.variant_type == VariantType.ENHANCED:
            base_conf += 0.08
        mean_confidence = max(0.1, min(0.98, base_conf))

        predictions: list[AnnotationPrediction] = []
        labels = ["vehicle", "person", "traffic_sign"]
        for index in range(1 + rng.randint(0, 2)):
            width = round(0.12 + rng.random() * 0.25, 2)
            height = round(0.12 + rng.random() * 0.22, 2)
            x = round(rng.random() * (1 - width), 2)
            y = round(rng.random() * (1 - height), 2)
            predictions.append(
                AnnotationPrediction(
                    class_name=labels[index % len(labels)],
                    confidence=max(0.1, min(0.99, mean_confidence - 0.08 + rng.random() * 0.16)),
                    box=BoundingBox(x=x, y=y, width=width, height=height),
                )
            )

        return AnnotationCandidate(
            variant_id=variant.id,
            provider=self.name,
            mean_confidence=round(mean_confidence, 3),
            class_consistency=round(max(0.1, min(0.99, 0.55 + quality_metrics.entropy_score * 0.25)), 3),
            box_stability=round(max(0.1, min(0.99, 0.45 + quality_metrics.sharpness_score * 0.30)), 3),
            cross_model_agreement=round(max(0.1, min(0.99, 0.50 + quality_metrics.contrast_std * 0.35)), 3),
            predictions=predictions,
        )


class HeuristicSyntheticProvider:
    name = "heuristic_synth"

    def plan(self, asset: Asset) -> SyntheticRecipe | None:
        scene = asset.scene_condition
        if scene is None:
            return None
        if scene.low_light_score > 0.45 or scene.backlight_score > 0.40:
            return SyntheticRecipe(
                asset_id=asset.id,
                name="targeted_recovery_mix",
                target_condition="low_light_backlight",
                operations=["copy_paste_small_objects", "contrast_aware_overlay", "rare_class_boost"],
                expected_gain=0.08,
            )
        return SyntheticRecipe(
            asset_id=asset.id,
            name="small_object_boost",
            target_condition="rare_small_objects",
            operations=["copy_paste_small_objects"],
            expected_gain=0.04,
        )

    def create_variant(self, asset: Asset, recipe: SyntheticRecipe) -> AssetVariant | None:
        source_path = Path(asset.original_file_path)
        if not source_path.exists():
            return None
        image = Image.open(source_path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
        h, w, _ = array.shape

        cx = max(10, w // 3)
        cy = max(10, h // 3)
        patch_w = max(12, w // 12)
        patch_h = max(12, h // 12)
        array[cy : cy + patch_h, cx : cx + patch_w, 0] = 255
        array[cy : cy + patch_h, cx : cx + patch_w, 1] = np.minimum(
            255, array[cy : cy + patch_h, cx : cx + patch_w, 1] + 80
        )

        synthetic = Image.fromarray(array)
        synthetic = ImageEnhance.Contrast(synthetic).enhance(1.05)
        synthetic_path = settings.variant_dir / f"{asset.id}_synthetic.png"
        synthetic.save(synthetic_path)

        return AssetVariant(
            asset_id=asset.id,
            variant_type=VariantType.SYNTHETIC,
            name=recipe.name,
            file_path=str(synthetic_path),
            transforms=recipe.operations,
            safety=TransformSafety.REVIEW_REQUIRED,
        )
