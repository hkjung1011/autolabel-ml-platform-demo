from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class WorkspaceArtifactPaths(BaseModel):
    workspace_root: str
    pair_manifest_path: str
    labeled_manifest_path: str
    split_manifest_path: str
    experiment_plan_path: str
    summary_path: str
    registration_report_dir: str


class PairGroup(BaseModel):
    key: str
    split: str
    frozen_split: str | None = None
    prefix: str
    shot_id: str
    exposures: dict[str, str] = Field(default_factory=dict)
    anchor_lux: str = "160"
    anchor_image_path: str | None = None
    anchor_label_path: str | None = None
    companion_luxes: list[str] = Field(default_factory=list)
    label_reusable: bool = False
    label_line_count: int = 0


class ExperimentArm(BaseModel):
    name: str
    stage: str
    description: str
    inputs: list[str] = Field(default_factory=list)
    expected_output: str


class ExperimentPlan(BaseModel):
    title: str
    phases: list[str] = Field(default_factory=list)
    arms: list[ExperimentArm] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchDatasetSummary(BaseModel):
    dataset_path: str
    weights_path: str | None = None
    total_images: int
    total_groups: int
    total_size_gb: float
    lux_counts: dict[str, int] = Field(default_factory=dict)
    combo_counts: dict[str, int] = Field(default_factory=dict)
    labeled_anchor_count: int
    labeled_with_40: int
    labeled_with_80: int
    labeled_with_both: int
    triple_group_count: int
    experiment_ready_count: int
    baseline_ready_count: int
    frozen_split_counts: dict[str, int] = Field(default_factory=dict)
    labeled_split_counts: dict[str, int] = Field(default_factory=dict)
    unmatched_image_count: int = 0
    unmatched_image_samples: list[str] = Field(default_factory=list)
    sample_groups: list[PairGroup] = Field(default_factory=list)
    artifact_paths: WorkspaceArtifactPaths


class BootstrapV1Request(BaseModel):
    dataset_path: str
    workspace_root: str = r"C:\paint_defect_research"
    weights_path: str | None = None
    materialize_workspace: bool = True

    def dataset_dir(self) -> Path:
        return Path(self.dataset_path)

    def workspace_dir(self) -> Path:
        return Path(self.workspace_root)


class BootstrapV1Response(BaseModel):
    summary: ResearchDatasetSummary
    experiment_plan: ExperimentPlan
    pair_groups: list[PairGroup] = Field(default_factory=list)
    message: str


class DatasetDiscoveryRequest(BaseModel):
    scan_root: str = r"D:\\"
    limit: int = 12
    min_images: int = 24


class LuxDatasetCandidate(BaseModel):
    dataset_root: str
    dataset_name: str
    images_root: str
    labels_root: str | None = None
    image_count: int
    group_count: int
    labeled_anchor_count: int
    lux_counts: dict[str, int] = Field(default_factory=dict)
    sample_image_path: str | None = None
    score: float
    notes: list[str] = Field(default_factory=list)


class DatasetDiscoveryResponse(BaseModel):
    scan_root: str
    candidates: list[LuxDatasetCandidate] = Field(default_factory=list)
    message: str


class StageCandidateRequest(BaseModel):
    source_dataset_root: str
    workspace_root: str = r"C:\paint_defect_research"
    staged_name: str | None = None
    max_groups: int = 48
    prefer_labeled: bool = True
    bootstrap_after_stage: bool = True


class StageCandidateResponse(BaseModel):
    source_dataset_root: str
    staged_dataset_root: str
    staged_workspace_root: str | None = None
    copied_images: int
    copied_labels: int
    selected_group_count: int
    selected_group_ids: list[str] = Field(default_factory=list)
    bootstrap_message: str | None = None


class RetinexRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    group_ids: list[str] = Field(default_factory=list)
    source_luxes: list[str] = Field(default_factory=lambda: ["80"])
    method: str = "msrcr"
    overwrite: bool = True
    params: dict[str, float | list[float] | int] = Field(default_factory=dict)


class RetinexVariantResult(BaseModel):
    group_id: str
    source_lux: str
    output_path: str
    anchor_path: str | None = None
    accepted: bool
    brightness_delta: float
    ssim_vs_anchor: float | None = None
    mean_intensity: float
    output_size_bytes: int


class RetinexJobResponse(BaseModel):
    job_id: str
    status: str
    method: str
    workspace_root: str
    total_groups: int
    processed_groups: int
    outputs: list[RetinexVariantResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    output_root: str


class RegistrationVerifyRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    variant_source: str = "retinex_msrcr"
    group_ids: list[str] = Field(default_factory=list)
    source_luxes: list[str] = Field(default_factory=lambda: ["80"])
    max_shift_px: int = 8
    max_corner_error_px: float = 2.5
    max_label_iou_drift: float = 0.08
    max_variant_shift_px: float = 2.0
    max_variant_iou_drift: float = 0.03
    min_similarity: float = 0.1
    materialize_accepted_dataset: bool = True


class RegistrationReport(BaseModel):
    group_id: str
    source_lux: str
    variant_source: str
    source_path: str
    variant_path: str
    anchor_path: str
    status: str
    similarity: float
    dx_px: int
    dy_px: int
    mean_corner_error_px: float
    label_iou_drift: float
    source_to_anchor_similarity: float
    source_to_anchor_dx_px: int
    source_to_anchor_dy_px: int
    variant_to_source_similarity: float
    variant_to_source_dx_px: int
    variant_to_source_dy_px: int
    variant_only_corner_error_px: float
    variant_only_iou_drift: float
    warp_matrix: list[list[float]]


class RegistrationJobResponse(BaseModel):
    job_id: str
    status: str
    variant_source: str
    workspace_root: str
    total_groups: int
    processed_groups: int
    accepted_count: int
    warning_count: int
    rejected_count: int
    reports: list[RegistrationReport] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report_path: str
    accepted_manifest_path: str | None = None
    registered_dataset_root: str | None = None


class FusionRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    group_ids: list[str] = Field(default_factory=list)
    source_luxes: list[str] = Field(default_factory=lambda: ["40", "80", "160"])
    method: str = "mertens"
    overwrite: bool = True
    use_labeled_only: bool = False
    emit_debug_artifacts: bool = True
    params: dict[str, float | int] = Field(default_factory=dict)


class FusionVariantResult(BaseModel):
    group_id: str
    source_luxes: list[str] = Field(default_factory=list)
    output_path: str
    weight_map_paths: dict[str, str] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    mean_intensity: float
    dynamic_range_score: float
    output_size_bytes: int


class FusionSkippedItem(BaseModel):
    group_id: str
    reason: str
    available_luxes: list[str] = Field(default_factory=list)
    requested_luxes: list[str] = Field(default_factory=list)
    required_min: int = 2


class FusionJobResponse(BaseModel):
    job_id: str
    status: str
    method: str
    workspace_root: str
    total_groups: int
    processed_groups: int
    outputs: list[FusionVariantResult] = Field(default_factory=list)
    skipped: list[FusionSkippedItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    output_root: str
    dataset_root: str | None = None


class TargetLuxRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    group_ids: list[str] = Field(default_factory=list)
    target_lux: int = 100
    source_luxes: list[str] = Field(default_factory=lambda: ["40", "80", "160"])
    overwrite: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


class TargetLuxVariantResult(BaseModel):
    group_id: str
    target_lux: int
    source_luxes_used: list[str] = Field(default_factory=list)
    blend_weights: dict[str, float] = Field(default_factory=dict)
    output_path: str
    mean_intensity: float
    dynamic_range_score: float
    output_size_bytes: int


class TargetLuxJobResponse(BaseModel):
    job_id: str
    status: str
    workspace_root: str
    target_lux: int
    total_groups: int
    processed_groups: int
    outputs: list[TargetLuxVariantResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    output_root: str


class ForensicWdrRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    group_ids: list[str] = Field(default_factory=list)
    source_luxes: list[str] = Field(default_factory=lambda: ["40", "80", "160"])
    overwrite: bool = True
    detail_boost: float = 1.35
    shadow_lift: float = 0.45
    highlight_protect: float = 0.55
    clahe_clip_limit: float = 2.2
    bilateral_sigma: float = 35.0


class ForensicWdrVariantResult(BaseModel):
    group_id: str
    source_luxes_used: list[str] = Field(default_factory=list)
    output_path: str
    mean_intensity: float
    dynamic_range_score: float
    local_contrast_score: float
    output_size_bytes: int


class ForensicWdrJobResponse(BaseModel):
    job_id: str
    status: str
    workspace_root: str
    total_groups: int
    processed_groups: int
    outputs: list[ForensicWdrVariantResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    output_root: str


class DynamicCaptureRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    target_mid_lux: int = 100
    preferred_source_luxes: list[str] = Field(default_factory=lambda: ["40", "80", "160"])
    max_preview_groups: int = 12


class DynamicCapturePlanItem(BaseModel):
    group_id: str
    split: str
    available_luxes: list[str] = Field(default_factory=list)
    capture_mode: str
    exposure_plan: list[str] = Field(default_factory=list)
    next_pixel_branch: str
    risk_level: str
    notes: list[str] = Field(default_factory=list)
    anchor_image_path: str | None = None


class DynamicCaptureResponse(BaseModel):
    workspace_root: str
    target_mid_lux: int
    recommended_global_mode: str
    total_groups: int
    bracket_ready_groups: int
    high_dynamic_range_groups: int
    forensic_followup_groups: int
    plans: list[DynamicCapturePlanItem] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class PixelMethodSummary(BaseModel):
    method_name: str
    implemented: bool
    status: str
    best_for: str
    summary: str
    output_count: int = 0


class PixelLabResponse(BaseModel):
    workspace_root: str
    recommended_method: str | None = None
    target_lux_ready: bool = False
    target_lux_value: int | None = None
    methods: list[PixelMethodSummary] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    preview_paths: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class EvaluationRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    include_arms: list[str] = Field(default_factory=lambda: ["raw160", "retinex", "mertens", "daf"])
    refresh_report: bool = True


class EvaluationArmSummary(BaseModel):
    arm_name: str
    dataset_root: str
    split_image_counts: dict[str, int] = Field(default_factory=dict)
    split_label_counts: dict[str, int] = Field(default_factory=dict)
    class_ids: list[int] = Field(default_factory=list)
    dataset_yaml_path: str | None = None
    ready: bool
    notes: list[str] = Field(default_factory=list)


class EvaluationRequirement(BaseModel):
    category: str
    name: str
    required: bool
    reason: str


class EvaluationReadinessReport(BaseModel):
    workspace_root: str
    completion_percent: int
    execution_readiness_percent: int
    phase_status: dict[str, str] = Field(default_factory=dict)
    arms: list[EvaluationArmSummary] = Field(default_factory=list)
    requirements: list[EvaluationRequirement] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class EvidenceRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    source_lux: str = "80"
    compare_arms: list[str] = Field(default_factory=lambda: ["raw80", "retinex80", "mertens", "daf"])
    group_ids: list[str] = Field(default_factory=list)
    refresh_report: bool = True


class EvidenceArmSummary(BaseModel):
    arm_name: str
    available_groups: int
    avg_dynamic_range: float
    avg_defect_visibility: float
    avg_background_suppression: float
    coverage_ratio: float | None = None
    evidence_score: float | None = None
    visibility_gain_vs_raw80: float | None = None
    dynamic_range_gain_vs_raw80: float | None = None
    notes: list[str] = Field(default_factory=list)


class EvidenceBenchmarkReport(BaseModel):
    workspace_root: str
    source_lux: str
    common_group_count: int
    arms: list[EvidenceArmSummary] = Field(default_factory=list)
    recommended_arm: str | None = None
    peak_arm: str | None = None
    key_takeaways: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class TrainingRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    arm: str = "raw160"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    weights_path: str = "yolov8n.pt"
    run_name: str | None = None
    dry_run: bool = True
    trainer_command: str | None = None


class TrainingRunResponse(BaseModel):
    workspace_root: str
    arm: str
    status: str
    dry_run: bool
    dataset_yaml_path: str | None = None
    output_dir: str
    run_name: str
    command_preview: str
    requested_device: str = "0"
    actual_device: str | None = None
    torch_cuda_available: bool | None = None
    runtime_mode: str | None = None
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class TrainingJobStatusResponse(BaseModel):
    job_id: str
    workspace_root: str
    arm: str
    status: str
    dry_run: bool
    run_name: str
    output_dir: str
    command_preview: str
    requested_device: str = "0"
    actual_device: str | None = None
    torch_cuda_available: bool | None = None
    runtime_mode: str | None = None
    dataset_yaml_path: str | None = None
    log_path: str | None = None
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    tail_lines: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class TrainingRunHistoryItem(BaseModel):
    run_name: str
    arm: str
    status: str
    dry_run: bool
    output_dir: str
    dataset_yaml_path: str | None = None
    command_preview: str | None = None
    requested_device: str | None = None
    actual_device: str | None = None
    torch_cuda_available: bool | None = None
    runtime_mode: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)


class TrainingRunHistoryResponse(BaseModel):
    workspace_root: str
    total_runs: int
    runs: list[TrainingRunHistoryItem] = Field(default_factory=list)


class TestEvaluationRequest(BaseModel):
    run_dir: str
    data_yaml_path: str | None = None
    device: str = "0"
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.6


class TestEvaluationPerClass(BaseModel):
    class_id: int
    name: str
    ap50: float
    ap50_95: float


class TestEvaluationResponse(BaseModel):
    run_dir: str
    status: str
    split: str = "test"
    model_path: str
    data_yaml_path: str
    output_dir: str
    requested_device: str = "0"
    actual_device: str | None = None
    torch_cuda_available: bool | None = None
    runtime_mode: str | None = None
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    per_class: list[TestEvaluationPerClass] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    results_json_path: str
    summary_text_path: str


class UsageScorecardResponse(BaseModel):
    workspace_root: str
    research_score: int
    field_score: int
    production_score: int
    export_ready: bool
    visual_trace_ready: bool
    recommended_arm: str | None = None
    latest_metric_summary: str | None = None
    strengths: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class CsvExportResponse(BaseModel):
    workspace_root: str
    export_root: str
    files: dict[str, str] = Field(default_factory=dict)
    message: str


class ArmComparisonRow(BaseModel):
    arm_name: str
    ready: bool
    available_groups: int = 0
    coverage_ratio: float | None = None
    evidence_score: float | None = None
    visibility_gain_vs_raw80: float | None = None
    dynamic_range_gain_vs_raw80: float | None = None
    latest_metric_name: str | None = None
    latest_metric_value: float | int | str | None = None
    latest_run_name: str | None = None
    status: str = "unknown"
    decision_tag: str = "review"
    notes: list[str] = Field(default_factory=list)


class ArmComparisonResponse(BaseModel):
    workspace_root: str
    deploy_candidate: str | None = None
    evidence_candidate: str | None = None
    training_candidate: str | None = None
    rows: list[ArmComparisonRow] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class PaperAblationRow(BaseModel):
    arm_name: str
    evidence_score: float | None = None
    visibility_gain_vs_raw80: float | None = None
    dynamic_range_gain_vs_raw80: float | None = None
    latest_metric_name: str | None = None
    latest_metric_value: float | int | str | None = None
    decision_tag: str = "review"
    interpretation: str = ""


class PaperPackResponse(BaseModel):
    workspace_root: str
    impact_domain: str
    paper_readiness_score: int
    working_title: str
    title_candidates: list[str] = Field(default_factory=list)
    novelty_statement: str
    target_problem: str
    abstract_draft: str
    contributions: list[str] = Field(default_factory=list)
    experiment_protocol: list[str] = Field(default_factory=list)
    ablation_rows: list[PaperAblationRow] = Field(default_factory=list)
    figure_checklist: list[str] = Field(default_factory=list)
    reproducibility_checklist: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_paper_actions: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str
    ablation_csv_path: str


class AccuracyAuditArm(BaseModel):
    arm_name: str
    label_mode: str
    split_image_counts: dict[str, int] = Field(default_factory=dict)
    evidence_score: float | None = None
    latest_metric_summary: str | None = None
    experiment_priority: int
    status: str
    rationale: list[str] = Field(default_factory=list)


class PixelMethodPlan(BaseModel):
    method_name: str
    implemented: bool
    readiness: str
    accuracy_hypothesis: str
    metrics_to_track: list[str] = Field(default_factory=list)
    next_step: str


class AccuracyAuditResponse(BaseModel):
    workspace_root: str
    current_label_mode: str
    detection_ready: bool
    segmentation_ready: bool
    segmentation_bootstrap_ready: bool = False
    segmentation_bootstrap_dataset_root: str | None = None
    segmentation_bootstrap_mode: str | None = None
    segmentation_bootstrap_refined_items: int = 0
    accuracy_readiness_score: int
    dataset_train_images: int
    dataset_val_images: int
    dataset_test_images: int
    baseline_arm: str | None = None
    baseline_metric_summary: str | None = None
    primary_eval_metrics: list[str] = Field(default_factory=list)
    first_experiments: list[str] = Field(default_factory=list)
    arms: list[AccuracyAuditArm] = Field(default_factory=list)
    pixel_methods: list[PixelMethodPlan] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class DataQualityIssue(BaseModel):
    category: str
    severity: str
    item_id: str
    summary: str
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class DataQualityAuditResponse(BaseModel):
    workspace_root: str
    registration_reports_scanned: int
    registered_groups_scanned: int
    position_issue_count: int
    severe_position_issue_count: int
    label_files_scanned: int
    label_issue_count: int
    invalid_label_count: int
    out_of_bounds_box_count: int
    tiny_box_count: int
    oversize_box_count: int
    duplicate_box_count: int
    suspect_group_ids: list[str] = Field(default_factory=list)
    suspect_label_files: list[str] = Field(default_factory=list)
    position_issues: list[DataQualityIssue] = Field(default_factory=list)
    label_issues: list[DataQualityIssue] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class ProgramStructureItem(BaseModel):
    module_name: str
    category: str
    status: str
    progress_percent: int
    summary: str
    blockers: list[str] = Field(default_factory=list)
    next_step: str


class ProgramStatusResponse(BaseModel):
    workspace_root: str
    overall_progress_percent: int
    execution_readiness_percent: int
    autolabel_progress_percent: int
    detection_progress_percent: int
    segmentation_progress_percent: int
    current_stage: str
    recommended_arm: str | None = None
    deploy_candidate: str | None = None
    commercial_stage: str | None = None
    commercial_readiness_score: int | None = None
    protected_source_count: int = 0
    staged_workspace_count: int = 0
    research_score: int
    field_score: int
    production_score: int
    summary_points: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    structure_items: list[ProgramStructureItem] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class AutoLabelBuildRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    include_arms: list[str] = Field(default_factory=lambda: ["raw160", "retinex", "mertens", "daf"])
    overwrite: bool = True
    focus_mode: str = "defect_only"
    include_lighting_anomalies: bool = False
    dark_threshold: int = 52
    bright_threshold: int = 228
    min_region_area_ratio: float = 0.012


class AutoLabelBuildResponse(BaseModel):
    workspace_root: str
    dataset_root: str
    dataset_yaml_path: str
    total_proposals: int
    focus_mode: str = "defect_only"
    class_names: list[str] = Field(default_factory=list)
    split_counts: dict[str, int] = Field(default_factory=dict)
    arm_counts: dict[str, int] = Field(default_factory=dict)
    proposal_modes: dict[str, int] = Field(default_factory=dict)
    anomaly_box_count: int = 0
    anomaly_image_count: int = 0
    anomaly_type_counts: dict[str, int] = Field(default_factory=dict)
    source_datasets: dict[str, str] = Field(default_factory=dict)
    report_json_path: str
    report_markdown_path: str
    message: str


class SegmentationBootstrapRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    source_dataset_name: str = "auto"
    include_splits: list[str] = Field(default_factory=lambda: ["train", "val", "test"])
    overwrite: bool = True
    padding_ratio: float = 0.02
    min_padding_px: int = 1
    refine_with_sam: bool = False
    sam_model: str = "mobile_sam.pt"
    sam_device: str = "auto"


class SegmentationBootstrapResponse(BaseModel):
    workspace_root: str
    source_dataset_name: str
    source_dataset_root: str
    dataset_root: str
    dataset_yaml_path: str
    mask_root: str
    total_items: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    class_ids: list[int] = Field(default_factory=list)
    sample_mask_paths: list[str] = Field(default_factory=list)
    bootstrap_mode: str = "coarse_box_mask"
    sam_used: bool = False
    sam_model: str | None = None
    sam_device: str | None = None
    refined_items: int = 0
    review_required_count: int = 0
    report_json_path: str
    report_markdown_path: str
    message: str


class ReviewQueueItem(BaseModel):
    proposal_id: str
    arm_name: str
    split: str
    proposal_mode: str
    review_status: str
    priority: str
    source_image_path: str = ""
    source_label_path: str = ""
    image_path: str
    label_path: str
    review_owner: str | None = None
    updated_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    review_history: list[str] = Field(default_factory=list)


class ReviewQueueResponse(BaseModel):
    workspace_root: str
    total_items: int
    reviewed_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    arm_counts: dict[str, int] = Field(default_factory=dict)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    items: list[ReviewQueueItem] = Field(default_factory=list)
    items_json_path: str
    report_json_path: str
    report_markdown_path: str
    approved_dataset_root: str | None = None
    approved_export_count: int = 0
    message: str


class ReviewQueueUpdateRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    proposal_id: str
    action: str
    review_owner: str | None = None
    note: str | None = None


class ReviewQueueExportResponse(BaseModel):
    workspace_root: str
    approved_dataset_root: str
    exported_items: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    items_json_path: str
    report_json_path: str
    report_markdown_path: str
    message: str


class DesktopPackagePlanResponse(BaseModel):
    workspace_root: str
    app_root: str
    entry_script_path: str
    build_script_path: str
    spec_path: str
    dist_dir: str
    exe_path: str
    exe_exists: bool
    pyinstaller_ready: bool
    build_ready: bool
    command_preview: str
    blockers: list[str] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class DesktopRuntimeCheckItem(BaseModel):
    check_name: str
    status: str
    summary: str
    detail: str | None = None
    action: str | None = None


class DesktopRuntimeCheckResponse(BaseModel):
    workspace_root: str
    readiness_score: int
    desktop_mode: str
    python_executable: str
    uv_available: bool
    pyinstaller_ready: bool
    ultralytics_available: bool
    opencv_available: bool
    torch_available: bool
    cuda_available: bool
    exe_exists: bool
    package_build_ready: bool
    gpu_summary: str
    checks: list[DesktopRuntimeCheckItem] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class LiveMonitorArtifact(BaseModel):
    label: str
    path: str
    category: str
    modified_at: str
    size_bytes: int | None = None


class LiveMonitorResponse(BaseModel):
    workspace_root: str
    source_dataset_path: str | None = None
    source_image_count: int = 0
    staged_image_count: int = 0
    retinex_output_count: int = 0
    forensic_wdr_output_count: int = 0
    registered_variant_count: int = 0
    mertens_output_count: int = 0
    daf_output_count: int = 0
    autolabel_proposal_count: int = 0
    autolabel_anomaly_box_count: int = 0
    autolabel_focus_mode: str | None = None
    segmentation_mask_count: int = 0
    segmentation_refined_items: int = 0
    reviewed_count: int = 0
    approved_count: int = 0
    latest_activity_at: str | None = None
    preview_paths: list[str] = Field(default_factory=list)
    recent_artifacts: list[LiveMonitorArtifact] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class OperatorGuideStep(BaseModel):
    step_name: str
    status: str
    progress_percent: int
    summary: str
    recommended_action: str
    ui_anchor: str
    api_hint: str | None = None


class OperatorGuideResponse(BaseModel):
    workspace_root: str
    operator_readiness_score: int
    current_focus: str
    primary_action: str
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    steps: list[OperatorGuideStep] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class SourceCatalogRequest(BaseModel):
    scan_root: str = r"D:\\"
    workspace_root: str = r"C:\paint_defect_research"
    limit: int = 24
    min_images: int = 24
    selected_dataset_roots: list[str] = Field(default_factory=list)


class SourceCatalogEntry(BaseModel):
    dataset_root: str
    dataset_name: str
    image_count: int
    group_count: int
    labeled_anchor_count: int
    lux_counts: dict[str, int] = Field(default_factory=dict)
    sample_image_path: str | None = None
    source_policy: str
    ingest_mode: str
    target_stage_name: str
    recommendation: str
    notes: list[str] = Field(default_factory=list)


class SourceCatalogResponse(BaseModel):
    scan_root: str
    workspace_root: str
    total_entries: int
    protected_source_count: int
    source_policy_summary: str
    entries: list[SourceCatalogEntry] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str
    message: str


class CommercialMilestone(BaseModel):
    milestone_name: str
    status: str
    progress_percent: int
    summary: str
    next_step: str


class CommercializationPlanRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    scan_root: str = r"D:\\"
    refresh_source_catalog: bool = True
    limit: int = 24
    min_images: int = 24


class CommercializationPlanResponse(BaseModel):
    workspace_root: str
    scan_root: str
    commercial_stage: str
    commercial_readiness_score: int
    protected_source_count: int
    staged_workspace_count: int
    source_catalog_path: str
    research_score: int
    field_score: int
    production_score: int
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    milestones: list[CommercialMilestone] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class CommercialStageRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    source_dataset_root: str
    staged_name: str | None = None
    max_groups: int = 96
    prefer_labeled: bool = True
    bootstrap_after_stage: bool = True
    run_pipeline_after_stage: bool = True
    include_training_plan: bool = True
    training_dry_run: bool = True


class CommercialStageResponse(BaseModel):
    workspace_root: str
    source_dataset_root: str
    source_policy: str
    ingest_mode: str
    staged_dataset_root: str
    staged_workspace_root: str | None = None
    copied_images: int
    copied_labels: int
    selected_group_count: int
    selected_group_ids: list[str] = Field(default_factory=list)
    bootstrap_message: str | None = None
    pipeline_report_path: str | None = None
    pipeline_markdown_path: str | None = None
    commercial_plan_path: str | None = None
    commercial_stage: str | None = None
    commercial_readiness_score: int | None = None
    message: str


class AblationRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    arms: list[str] = Field(default_factory=lambda: ["raw160", "retinex", "mertens", "daf"])
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    weights_path: str = "yolov8n.pt"
    dry_run: bool = True


class AblationRunResponse(BaseModel):
    workspace_root: str
    status: str
    dry_run: bool
    runs: list[TrainingRunResponse] = Field(default_factory=list)
    report_json_path: str
    report_markdown_path: str


class WorkspacePipelineRunRequest(BaseModel):
    workspace_root: str = r"C:\paint_defect_research"
    retinex_luxes: list[str] = Field(default_factory=lambda: ["80"])
    fusion_luxes: list[str] = Field(default_factory=lambda: ["40", "80", "160"])
    evidence_source_lux: str = "80"
    evidence_compare_arms: list[str] = Field(default_factory=lambda: ["raw80", "retinex80", "mertens", "daf"])
    include_training_plan: bool = True
    training_arm: str = "raw160"
    training_dry_run: bool = True


class WorkspacePipelineStage(BaseModel):
    stage_name: str
    status: str
    summary: str
    artifact_path: str | None = None
    preview_paths: list[str] = Field(default_factory=list)


class WorkspacePipelineRunResponse(BaseModel):
    workspace_root: str
    status: str
    completion_percent: int
    execution_readiness_percent: int
    recommended_arm: str | None = None
    peak_arm: str | None = None
    stages: list[WorkspacePipelineStage] = Field(default_factory=list)
    report_json_path: str | None = None
    report_markdown_path: str | None = None
