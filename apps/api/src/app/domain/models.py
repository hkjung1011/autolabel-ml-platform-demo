from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class TransformSafety(str, Enum):
    LABEL_SAFE = "label_safe"
    REVIEW_REQUIRED = "review_required"


class VariantType(str, Enum):
    ORIGINAL = "original"
    ENHANCED = "enhanced"
    SYNTHETIC = "synthetic"


class AssetKind(str, Enum):
    RGB = "rgb"
    RAW = "raw"
    MULTI_EXPOSURE = "multi_exposure"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


class SceneCondition(BaseModel):
    low_light_score: float
    backlight_score: float
    blur_score: float
    noise_score: float
    highlight_clipping_score: float
    shadow_crushing_score: float
    recommended_chain: list[str] = Field(default_factory=list)
    summary: str


class QualityMetric(BaseModel):
    brightness_mean: float
    contrast_std: float
    sharpness_score: float
    entropy_score: float


class ExposureLightingDecision(BaseModel):
    control_mode: str
    target_exposure_bias: float
    recommended_gain: float
    recommended_shutter_ratio: float
    lighting_action: str
    vision_ready_score: float
    rationale: str


class ArtifactMetric(BaseModel):
    halo_risk: float
    color_shift_risk: float
    oversharpen_risk: float
    noise_amplification_risk: float


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class AnnotationPrediction(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox


class AnnotationCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    variant_id: str
    provider: str
    mean_confidence: float
    class_consistency: float
    box_stability: float
    cross_model_agreement: float
    predictions: list[AnnotationPrediction] = Field(default_factory=list)


class AnnotationReview(BaseModel):
    id: str = Field(default_factory=lambda: new_id("review"))
    candidate_id: str
    required: bool
    reason: str
    created_at: datetime = Field(default_factory=now_utc)


class AssetVariant(BaseModel):
    id: str = Field(default_factory=lambda: new_id("var"))
    asset_id: str
    variant_type: VariantType
    name: str
    file_path: str
    transforms: list[str] = Field(default_factory=list)
    safety: TransformSafety = TransformSafety.LABEL_SAFE
    quality_metrics: QualityMetric | None = None
    artifact_metrics: ArtifactMetric | None = None


class Asset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("asset"))
    dataset_id: str = "dataset_demo"
    name: str
    kind: AssetKind = AssetKind.RGB
    original_file_path: str
    created_at: datetime = Field(default_factory=now_utc)
    tags: list[str] = Field(default_factory=list)
    scene_condition: SceneCondition | None = None
    exposure_lighting_decision: ExposureLightingDecision | None = None
    variants: list[AssetVariant] = Field(default_factory=list)
    label_candidates: list[AnnotationCandidate] = Field(default_factory=list)
    reviews: list[AnnotationReview] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    synthetic_variant_ids: list[str] = Field(default_factory=list)


class SyntheticRecipe(BaseModel):
    id: str = Field(default_factory=lambda: new_id("syn"))
    asset_id: str
    name: str
    target_condition: str
    operations: list[str] = Field(default_factory=list)
    expected_gain: float


class SyntheticAssetLink(BaseModel):
    id: str = Field(default_factory=lambda: new_id("slink"))
    source_asset_id: str
    synthetic_variant_id: str
    recipe_id: str


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    asset_id: str
    status: JobStatus = JobStatus.PENDING
    job_type: str
    created_at: datetime = Field(default_factory=now_utc)
    finished_at: datetime | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("eval"))
    mode: str
    map50: float
    mean_iou: float
    label_trust_score: float


class SubsetBenchmark(BaseModel):
    id: str = Field(default_factory=lambda: new_id("bench"))
    subset_name: str
    baseline_score: float
    improved_score: float
    delta: float


class AuditLog(BaseModel):
    id: str = Field(default_factory=lambda: new_id("audit"))
    action: str
    entity_type: str
    entity_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)


class DashboardSummary(BaseModel):
    total_assets: int
    low_light_assets: int
    review_queue_count: int
    data_salvage_rate: float
    condition_gain: float
    label_trust_score: float
    synthetic_gain_score: float
    subset_benchmarks: list[SubsetBenchmark] = Field(default_factory=list)


class AssetListResponse(BaseModel):
    items: list[Asset]


class UploadAssetResponse(BaseModel):
    asset: Asset
    message: str


class PipelineRunResponse(BaseModel):
    asset: Asset
    job: Job
    evaluation: EvaluationResult
    synthetic_recipe: SyntheticRecipe | None


class DemoSeedResponse(BaseModel):
    assets: list[Asset]
    message: str
