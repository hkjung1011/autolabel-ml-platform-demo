from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CANONICAL_CLASS_NAMES = {
    0: "corrosion",
    1: "crack",
    2: "coating_damage",
    3: "scratch",
    4: "dent_deformation",
    5: "hole_pit",
    6: "weld_defect",
    7: "blister_bubble",
    8: "contamination",
    9: "edge_damage",
    10: "burr",
    11: "surface_stain",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CANONICAL_CLASS_NAMES.items()}

DOMAIN_ALLOWED_CLASS_IDS = {
    "ship_defect": [0, 1, 2, 4, 5, 6, 7, 8, 9],
    "metal_plate_defect": [0, 1, 3, 4, 5, 6, 8, 9, 10, 11],
}


class DefectAutolabelProjectRequest(BaseModel):
    input_root: str
    workspace_root: str
    domain: Literal["ship_defect", "metal_plate_defect"]
    dataset_mode: Literal["paired_lux", "single_image"] = "paired_lux"
    split_strategy: Literal["folder", "manifest", "random"] = "folder"
    debug: bool = False


class DefectAssetRecord(BaseModel):
    asset_id: str
    image_path: str
    group_id: str | None = None
    shot_id: str | None = None
    reported_lux: int | None = None
    estimated_lux_bucket: str
    split: str = "train"
    width: int
    height: int


class DefectQualityMetrics(BaseModel):
    asset_id: str
    mean_luma: float
    median_luma: float
    std_luma: float
    shadow_clip_ratio: float
    highlight_clip_ratio: float
    local_contrast_score: float
    laplacian_blur_score: float
    specular_glare_score: float
    vision_ready_score: float
    lux_bucket: str


class DefectGroupRecord(BaseModel):
    group_id: str
    member_asset_ids: list[str] = Field(default_factory=list)
    anchor_asset_id: str | None = None
    group_mode: str = "single"
    available_luxes: list[int] = Field(default_factory=list)


class DefectProposal(BaseModel):
    proposal_id: str
    asset_id: str
    image_path: str
    split: str
    image_width: int
    image_height: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_yolo: list[float]
    source_mode: str
    views_supporting: list[str] = Field(default_factory=list)
    review_required: bool
    priority: str
    quality_flags: list[str] = Field(default_factory=list)
    lux_bucket: str


class DefectReviewItem(BaseModel):
    proposal_id: str
    asset_id: str
    image_path: str
    split: str
    image_width: int
    image_height: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_yolo: list[float]
    source_mode: str
    views_supporting: list[str] = Field(default_factory=list)
    lux_bucket: str
    priority: str
    quality_flags: list[str] = Field(default_factory=list)
    review_status: str
    review_owner: str | None = None
    updated_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    review_history: list[str] = Field(default_factory=list)


class DefectAutolabelProjectResponse(BaseModel):
    workspace_root: str
    domain: str
    total_assets: int
    total_groups: int
    asset_manifest_path: str
    group_manifest_path: str
    quality_manifest_path: str
    report_json_path: str
    report_markdown_path: str
    message: str


class DefectAutolabelRunRequest(BaseModel):
    workspace_root: str
    domain: Literal["ship_defect", "metal_plate_defect"]
    run_mode: Literal["full", "detect_only", "propagate_only"] = "full"
    overwrite: bool = True
    debug: bool = False


class DefectAutolabelRunResponse(BaseModel):
    workspace_root: str
    domain: str
    total_assets: int
    grouped_assets: int
    total_groups: int
    anchor_selected_groups: int
    proposal_count: int
    review_required_count: int
    exported_count: int = 0
    manifest_path: str
    review_queue_path: str
    report_json_path: str
    report_markdown_path: str
    message: str


class DefectReviewBuildResponse(BaseModel):
    workspace_root: str
    total_items: int
    reviewed_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    items: list[DefectReviewItem] = Field(default_factory=list)
    items_json_path: str
    report_json_path: str
    report_markdown_path: str
    message: str


class DefectReviewUpdateRequest(BaseModel):
    workspace_root: str
    proposal_id: str
    action: Literal["approve", "reject", "needs_edit", "reset"]
    review_owner: str | None = None
    note: str | None = None


class DefectExportRequest(BaseModel):
    workspace_root: str
    domain: Literal["ship_defect", "metal_plate_defect"]
    overwrite: bool = True


class DefectExportResponse(BaseModel):
    workspace_root: str
    domain: str
    export_root: str
    dataset_yaml_path: str
    exported_images: int
    exported_labels: int
    accepted_proposals: int
    report_json_path: str
    report_markdown_path: str
    message: str
