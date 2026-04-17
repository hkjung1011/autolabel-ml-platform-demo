from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.domain.models import (
    AnnotationCandidate,
    ArtifactMetric,
    Asset,
    AssetKind,
    AssetVariant,
    ExposureLightingDecision,
    QualityMetric,
    SceneCondition,
    SyntheticRecipe,
)


class SceneAnalyzerPlugin(Protocol):
    name: str

    def analyze(self, image_path: Path, kind: AssetKind) -> tuple[SceneCondition, QualityMetric]:
        ...


class EnhancementProvider(Protocol):
    name: str

    def create_variant(
        self,
        asset: Asset,
        image_path: Path,
        scene_condition: SceneCondition,
    ) -> tuple[AssetVariant, QualityMetric, ArtifactMetric]:
        ...


class ExposureLightingController(Protocol):
    name: str

    def plan(
        self,
        scene_condition: SceneCondition,
        quality_metrics: QualityMetric,
    ) -> ExposureLightingDecision:
        ...


class AutoLabelProvider(Protocol):
    name: str

    def infer(self, variant: AssetVariant, quality_metrics: QualityMetric) -> AnnotationCandidate:
        ...


class SyntheticDataProvider(Protocol):
    name: str

    def plan(self, asset: Asset) -> SyntheticRecipe | None:
        ...

    def create_variant(self, asset: Asset, recipe: SyntheticRecipe) -> AssetVariant | None:
        ...
