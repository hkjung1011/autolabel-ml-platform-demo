from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.config import settings
from app.domain.models import (
    AnnotationReview,
    Asset,
    AssetKind,
    AssetListResponse,
    AssetVariant,
    DashboardSummary,
    DemoSeedResponse,
    EvaluationResult,
    Job,
    JobStatus,
    PipelineRunResponse,
    SceneCondition,
    SyntheticAssetLink,
    TransformSafety,
    VariantType,
)
from app.plugins.demo_plugins import (
    HeuristicAutoLabelProvider,
    HeuristicEnhancementProvider,
    HeuristicExposureLightingController,
    HeuristicSceneAnalyzer,
    HeuristicSyntheticProvider,
)
from app.services.repository import repository


class DemoPipelineService:
    def __init__(self) -> None:
        self.scene_analyzer = HeuristicSceneAnalyzer()
        self.exposure_controller = HeuristicExposureLightingController()
        self.enhancer = HeuristicEnhancementProvider()
        self.labeler = HeuristicAutoLabelProvider()
        self.synthetic_provider = HeuristicSyntheticProvider()

    def seed_demo_assets(self) -> DemoSeedResponse:
        assets: list[Asset] = []
        for name, palette in [
            ("night_street", (24, 28, 42)),
            ("backlight_crossing", (40, 48, 54)),
            ("warehouse_low_light", (30, 35, 30)),
        ]:
            image_path = settings.upload_dir / f"{name}.png"
            if not image_path.exists():
                self._generate_demo_image(image_path, name, palette)
            asset = Asset(name=name, kind=AssetKind.RGB, original_file_path=str(image_path), tags=["demo"])
            self._attach_original_variant(asset)
            repository.add_asset(asset)
            repository.log("seeded", "asset", asset.id, {"name": asset.name})
            assets.append(asset)
        return DemoSeedResponse(assets=assets, message="Demo assets created.")

    def create_asset_from_upload(self, file_path: Path, name: str, kind: AssetKind = AssetKind.RGB) -> Asset:
        asset = Asset(name=name, kind=kind, original_file_path=str(file_path))
        self._attach_original_variant(asset)
        repository.add_asset(asset)
        repository.log("uploaded", "asset", asset.id, {"path": str(file_path)})
        return asset

    def list_assets(self) -> AssetListResponse:
        return AssetListResponse(items=repository.list_assets())

    def get_asset(self, asset_id: str) -> Asset:
        return repository.get_asset(asset_id)

    def run_pipeline(self, asset_id: str) -> PipelineRunResponse:
        asset = repository.get_asset(asset_id)
        job = Job(asset_id=asset.id, job_type="pixel_intelligence_pipeline", status=JobStatus.RUNNING)
        repository.save_job(job)

        image_path = Path(asset.original_file_path)
        scene, original_quality = self.scene_analyzer.analyze(image_path, asset.kind)
        asset.scene_condition = scene
        asset.exposure_lighting_decision = self.exposure_controller.plan(scene, original_quality)

        original_variant = next(variant for variant in asset.variants if variant.variant_type == VariantType.ORIGINAL)
        original_variant.quality_metrics = original_quality
        original_variant.artifact_metrics = None

        enhanced_variant, _, _ = self.enhancer.create_variant(asset, image_path, scene)
        asset.variants = [original_variant, enhanced_variant]

        asset.label_candidates = [
            self.labeler.infer(original_variant, original_variant.quality_metrics or original_quality),
            self.labeler.infer(enhanced_variant, enhanced_variant.quality_metrics or original_quality),
        ]
        selected = self._select_best_candidate(asset)
        asset.selected_candidate_id = selected.id

        asset.reviews = []
        if enhanced_variant.safety == TransformSafety.REVIEW_REQUIRED:
            asset.reviews.append(
                AnnotationReview(
                    candidate_id=selected.id,
                    required=True,
                    reason="Enhanced variant includes review-required transforms.",
                )
            )

        recipe = self.synthetic_provider.plan(asset)
        if recipe is not None:
            repository.save_recipe(recipe)
            synthetic_variant = self.synthetic_provider.create_variant(asset, recipe)
            if synthetic_variant is not None:
                asset.synthetic_variant_ids = [synthetic_variant.id]
                asset.variants.append(synthetic_variant)
                link = SyntheticAssetLink(
                    source_asset_id=asset.id,
                    synthetic_variant_id=synthetic_variant.id,
                    recipe_id=recipe.id,
                )
                repository.save_link(link)

        evaluation = self._build_evaluation(asset, scene, selected.mean_confidence)
        repository.save_evaluation(evaluation)
        for benchmark in self._build_benchmarks(asset):
            repository.save_benchmark(benchmark)

        job.status = JobStatus.COMPLETED
        job.finished_at = datetime.now(UTC)
        job.output = {"selected_candidate_id": asset.selected_candidate_id}
        repository.log("pipeline_completed", "asset", asset.id, {"selected_candidate_id": asset.selected_candidate_id})
        return PipelineRunResponse(asset=asset, job=job, evaluation=evaluation, synthetic_recipe=recipe)

    def dashboard_summary(self) -> DashboardSummary:
        assets = repository.list_assets()
        total_assets = len(assets)
        low_light_assets = sum(
            1 for asset in assets if asset.scene_condition and asset.scene_condition.low_light_score > 0.45
        )
        review_queue_count = sum(len(asset.reviews) for asset in assets)
        selected_candidates = [
            next((candidate for candidate in asset.label_candidates if candidate.id == asset.selected_candidate_id), None)
            for asset in assets
        ]
        selected_candidates = [candidate for candidate in selected_candidates if candidate is not None]
        label_trust_score = round(
            sum(candidate.mean_confidence for candidate in selected_candidates) / max(1, len(selected_candidates)),
            3,
        )
        data_salvage_rate = round(
            sum(1 for asset in assets if asset.selected_candidate_id is not None) / max(1, total_assets),
            3,
        )
        subset_benchmarks = sorted(repository.benchmarks.values(), key=lambda item: item.subset_name)
        condition_gain = round(
            sum(benchmark.delta for benchmark in subset_benchmarks) / max(1, len(subset_benchmarks)),
            3,
        )
        synthetic_gain_score = round(
            sum(recipe.expected_gain for recipe in repository.recipes.values()) / max(1, len(repository.recipes)),
            3,
        )
        return DashboardSummary(
            total_assets=total_assets,
            low_light_assets=low_light_assets,
            review_queue_count=review_queue_count,
            data_salvage_rate=data_salvage_rate,
            condition_gain=condition_gain,
            label_trust_score=label_trust_score,
            synthetic_gain_score=synthetic_gain_score,
            subset_benchmarks=subset_benchmarks,
        )

    def _attach_original_variant(self, asset: Asset) -> None:
        asset.variants = [
            AssetVariant(
                asset_id=asset.id,
                variant_type=VariantType.ORIGINAL,
                name="original",
                file_path=asset.original_file_path,
                transforms=["preserve_original"],
                safety=TransformSafety.LABEL_SAFE,
            )
        ]

    def _select_best_candidate(self, asset: Asset):
        def score(candidate):
            return (
                candidate.mean_confidence * 0.45
                + candidate.class_consistency * 0.20
                + candidate.box_stability * 0.20
                + candidate.cross_model_agreement * 0.15
            )

        return max(asset.label_candidates, key=score)

    def _build_evaluation(self, asset: Asset, scene: SceneCondition, confidence: float) -> EvaluationResult:
        base = 0.42 + confidence * 0.35
        penalty = scene.blur_score * 0.08 + scene.noise_score * 0.05
        map50 = round(max(0.2, min(0.97, base - penalty)), 3)
        mean_iou = round(max(0.2, min(0.95, 0.38 + confidence * 0.33 - scene.blur_score * 0.05)), 3)
        label_trust = round(max(0.2, min(0.98, confidence * 0.92)), 3)
        return EvaluationResult(mode="mixed", map50=map50, mean_iou=mean_iou, label_trust_score=label_trust)

    def _build_benchmarks(self, asset: Asset):
        scene = asset.scene_condition or SceneCondition(
            low_light_score=0.0,
            backlight_score=0.0,
            blur_score=0.0,
            noise_score=0.0,
            highlight_clipping_score=0.0,
            shadow_crushing_score=0.0,
            recommended_chain=["preserve_original"],
            summary="N/A",
        )
        return [
            self._benchmark("low_light", scene.low_light_score, bonus=0.14),
            self._benchmark("backlight", scene.backlight_score, bonus=0.11),
            self._benchmark("noise_heavy", scene.noise_score, bonus=0.09),
        ]

    def _benchmark(self, subset_name: str, severity: float, bonus: float):
        baseline = round(max(0.18, 0.52 - severity * 0.22), 3)
        improved = round(min(0.95, baseline + bonus * max(0.25, severity + 0.2)), 3)
        from app.domain.models import SubsetBenchmark

        return SubsetBenchmark(
            subset_name=subset_name,
            baseline_score=baseline,
            improved_score=improved,
            delta=round(improved - baseline, 3),
        )

    def _generate_demo_image(self, path: Path, name: str, base_rgb: tuple[int, int, int]) -> None:
        width, height = 960, 540
        image = Image.new("RGB", (width, height), base_rgb)
        draw = ImageDraw.Draw(image)

        if "backlight" in name:
            for radius in range(180, 0, -8):
                alpha = 255 - radius
                color = tuple(min(255, channel + alpha) for channel in (80, 70, 60))
                draw.ellipse(
                    (width - 280 - radius, 80 - radius, width - 280 + radius, 80 + radius),
                    fill=color,
                )
        else:
            draw.rectangle((0, 0, width, height // 2), fill=tuple(min(255, channel + 10) for channel in base_rgb))

        for idx, x in enumerate([180, 340, 500, 720]):
            y = 320 if idx % 2 == 0 else 280
            draw.rectangle((x, y, x + 80, y + 140), outline=(220, 220, 220), width=3)
            draw.rectangle((x + 18, y + 30, x + 42, y + 58), fill=(255, 210, 80))

        draw.rectangle((0, 430, width, height), fill=(45, 45, 52))
        draw.text((30, 26), name, fill=(255, 255, 255))
        image.save(path)


pipeline_service = DemoPipelineService()
