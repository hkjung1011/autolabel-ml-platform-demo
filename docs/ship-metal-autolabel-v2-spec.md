# Original-Only Ship / Metal Defect Auto-Labeling V2

## 1. Product decision

This V2 is a narrow product, not a research platform.

It only does four things:

1. Read original inspection images.
2. Generate defect auto-label proposals for ship and metal-plate inspections.
3. Route uncertain proposals to a review queue.
4. Export reviewed labels as a detector-training dataset.

It does not do these in V2:

- capture-device control
- exposure recommendation UI
- Retinex, fusion, or restored-image export
- synthetic data generation
- training orchestration
- benchmarking dashboards
- paper/commercial packaging

The main domain difference is lighting level. Images may represent the same shot under different `lux` conditions, but the program must still treat the original image as the source of truth.

## 2. Core operating rule

The system must never replace the original image with an enhanced image in the final dataset.

Allowed:

- in-memory normalization for inference only
- temporary debug artifacts when `debug=true`
- label propagation from one original frame to another original frame in the same lux group

Not allowed:

- exporting enhanced images as if they were original inspection evidence
- mixing synthetic or fused images into the training export of this product

## 3. Supported inspection domains

The operator must choose one domain at run time:

- `ship_defect`
- `metal_plate_defect`

Each domain uses a constrained label subset so that false positives from unrelated classes are suppressed.

### 3.1 Canonical class set

Use one canonical label map internally:

| class_id | class_name        | ship_defect | metal_plate_defect |
| --- | --- | --- | --- |
| 0 | corrosion         | yes | yes |
| 1 | crack             | yes | yes |
| 2 | coating_damage    | yes | no  |
| 3 | scratch           | no  | yes |
| 4 | dent_deformation  | yes | yes |
| 5 | hole_pit          | yes | yes |
| 6 | weld_defect       | yes | yes |
| 7 | blister_bubble    | yes | no  |
| 8 | contamination     | yes | yes |
| 9 | edge_damage       | yes | yes |
| 10 | burr             | no  | yes |
| 11 | surface_stain    | no  | yes |

Rules:

- `ship_defect` allows classes `0,1,2,4,5,6,7,8,9`
- `metal_plate_defect` allows classes `0,1,3,4,5,6,8,9,10,11`
- proposals outside the selected domain subset are discarded before review-queue creation

### 3.2 Annotation scope

V2 uses `bbox` as the only production label type.

Reason:

- faster review throughput
- simpler propagation across lux groups
- lower implementation risk than polygon masks

Segmentation can be added later, but it is out of scope for this product spec.

## 4. Input contract

### 4.1 Accepted files

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

### 4.2 Dataset modes

The system must support two modes:

1. `paired_lux`
2. `single_image`

`paired_lux` is the default and preferred mode.

### 4.3 Paired lux dataset assumption

In `paired_lux` mode, multiple original images may represent the same shot under different lux values.

Expected examples:

- `panelA_shot003_lux40.jpg`
- `panelA_shot003_lux80.jpg`
- `panelA_shot003_lux160.jpg`

Required parsed fields:

- `group_id`
- `shot_id`
- `reported_lux` if present in filename or manifest
- `domain`
- `image_path`

If lux is not available in the filename, the system may read it from:

- a CSV manifest
- EXIF/custom metadata
- folder name such as `lux40/`, `lux80/`, `lux160/`

### 4.4 Single-image fallback

If no group can be formed, the image is processed independently.

No propagation is attempted in single-image mode.

## 5. Minimal operator flow

The desktop or web UI should expose only this flow:

1. Select input folder.
2. Select domain: `ship_defect` or `metal_plate_defect`.
3. Choose dataset mode: `paired_lux` or `single_image`.
4. Choose export format: `yolo_detection`.
5. Run auto-labeling.
6. Review uncertain proposals.
7. Export reviewed dataset.

UI panels that belong to the existing research pipeline should be hidden for this mode.

## 6. Processing pipeline

### 6.1 Stage A: ingest and grouping

For every image:

1. compute `asset_id`
2. parse `group_id`, `shot_id`, and `reported_lux`
3. classify the image into a `lux_bucket`
4. save an `asset_manifest` entry

Output artifact:

- `manifests/asset_manifest.json`

### 6.2 Stage B: image quality and lux analysis

For every image, compute:

- `mean_luma`
- `median_luma`
- `std_luma`
- `highlight_clip_ratio`
- `shadow_clip_ratio`
- `local_contrast_score`
- `laplacian_blur_score`
- `specular_glare_score`
- `vision_ready_score`

Recommended formulas:

- `shadow_clip_ratio = pixels(luma <= 12) / total_pixels`
- `highlight_clip_ratio = pixels(luma >= 243) / total_pixels`
- `laplacian_blur_score = min(var(Laplacian(gray)) / 250.0, 1.0)`
- `exposure_center_score = 1.0 - min(abs(mean_luma - 128.0) / 128.0, 1.0)`
- `vision_ready_score = 0.30 * exposure_center_score + 0.25 * local_contrast_score + 0.20 * laplacian_blur_score + 0.15 * (1.0 - shadow_clip_ratio) + 0.10 * (1.0 - highlight_clip_ratio)`

Lux buckets:

- `very_dark`: `mean_luma < 45`
- `dark`: `45 <= mean_luma < 80`
- `normal`: `80 <= mean_luma < 170`
- `bright`: `170 <= mean_luma < 220`
- `glare`: `mean_luma >= 220` or `highlight_clip_ratio >= 0.08`

Output artifact:

- `analyses/quality/<asset_id>.json`

### 6.3 Stage C: anchor selection for paired lux groups

If a group contains multiple original images, choose one `anchor_image`.

Anchor selection is not "highest lux wins".

Use:

- highest `vision_ready_score`
- penalize `glare`
- penalize severe blur
- prefer images with valid geometric overlap against the rest of the group

Recommended tie breaker order:

1. highest `vision_ready_score`
2. lowest `specular_glare_score`
3. highest `reported_lux`
4. lexicographically smallest path

Output artifact:

- `manifests/group_manifest.json`

Each group entry must contain:

- `group_id`
- `member_asset_ids`
- `anchor_asset_id`
- `group_mode`
- `available_luxes`

### 6.4 Stage D: inference views

The model runs on the original image plus in-memory derived views.

Required views:

1. `original`
2. `normalized_luma`
3. `edge_emphasis`

View rules:

- `original`: the untouched source image
- `normalized_luma`: CLAHE or gamma-adjusted copy in memory only
- `edge_emphasis`: unsharp/high-pass guided view in memory only for crack and scratch sensitivity

No derived view is allowed into the exported dataset.

### 6.5 Stage E: detection

Use a domain-specific detector per domain:

- `weights/ship_defect_detector.pt`
- `weights/metal_plate_defect_detector.pt`

Inference procedure:

1. run detection on each view
2. map outputs back to original-image coordinates
3. fuse proposals with Weighted Box Fusion or Soft-NMS
4. apply domain class whitelist
5. attach view-consistency metadata

Each proposal must contain:

- `asset_id`
- `source_mode`: `direct_detect`
- `class_id`
- `confidence`
- `bbox_xyxy`
- `views_supporting`
- `lux_bucket`
- `quality_flags`

### 6.6 Stage F: paired-lux propagation

If a group has multiple original images:

1. run full detection on the `anchor_image`
2. estimate transform from anchor to companion with the existing registration utilities
3. propagate anchor boxes to the companion image
4. run lightweight direct detection on the companion image
5. merge propagated boxes and direct boxes
6. reject propagation if drift or similarity checks fail

Propagation acceptance thresholds:

- `max_shift_px <= 8`
- `mean_corner_error_px <= 2.5`
- `variant_only_iou_drift <= 0.03`
- `source_to_anchor_similarity >= 0.10`

Proposal sources after merge:

- `anchor_propagated`
- `direct_detect`
- `direct_plus_propagated`

If registration fails, the system must fall back to direct detection only.

### 6.7 Stage G: confidence calibration and review routing

All proposals are written, but review priority depends on risk.

Base detection thresholds:

- `very_dark`: `0.18`
- `dark`: `0.22`
- `normal`: `0.28`
- `bright`: `0.30`
- `glare`: `0.32`

Hard review triggers:

- `lux_bucket` is `very_dark` or `glare`
- proposal source includes propagation
- overlapping boxes of different classes
- box touches image boundary
- box area ratio `< 0.001`
- image blur score `< 0.20`
- confidence gap between top-2 classes `< 0.10`

Priority policy:

- `high`: glare, very dark, propagation, test split
- `medium`: dark, bright, boundary-touching, low confidence
- `normal`: normal-light direct detections with strong multi-view support

Output artifacts:

- `predictions/raw/<asset_id>.json`
- `predictions/fused/<asset_id>.json`
- `review_queue/items.json`

### 6.8 Stage H: export

Only original image paths are exported.

Export format for V2:

- YOLO detection labels

Directory layout:

```text
workspace/
  exports/
    yolo_detection/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      meta/
        dataset.yaml
        export_manifest.json
```

Each exported label file must correspond to the original image file name.

## 7. Required internal data models

The existing `research_models.py` file is already large. V2 should add a dedicated module:

- `apps/api/src/app/domain/defect_autolabel_models.py`

Required models:

```python
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


class DefectProposal(BaseModel):
    proposal_id: str
    asset_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_yolo: list[float]
    source_mode: str
    views_supporting: list[str]
    review_required: bool
    priority: str
    quality_flags: list[str]


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
```

## 8. Required service modules

V2 should be implemented as a separate narrow path, not by expanding the current research services again.

Add these modules:

- `apps/api/src/app/services/defect_dataset_ingest.py`
- `apps/api/src/app/services/defect_quality.py`
- `apps/api/src/app/services/defect_grouping.py`
- `apps/api/src/app/services/defect_anchor_selector.py`
- `apps/api/src/app/services/defect_inference.py`
- `apps/api/src/app/services/defect_propagation.py`
- `apps/api/src/app/services/defect_review.py`
- `apps/api/src/app/services/defect_export.py`
- `apps/api/src/app/services/defect_autolabel_runner.py`

Reuse without duplicating:

- `plugins/registration/translation_aligner.py`
- `plugins/registration/verifier.py`
- atomic file writers in `core/atomic_io.py`

Do not reuse these in the V2 product flow:

- Retinex runners
- fusion runners
- commercialization/reporting scorecards
- paper-pack logic

## 9. API contract

Add a focused V2 namespace:

- `POST /api/research/v2/defect-autolabel/project/init`
- `POST /api/research/v2/defect-autolabel/run`
- `GET /api/research/v2/defect-autolabel/latest`
- `POST /api/research/v2/defect-autolabel/review/build`
- `POST /api/research/v2/defect-autolabel/review/update`
- `POST /api/research/v2/defect-autolabel/export`

### 9.1 `project/init`

Purpose:

- scan originals
- build asset manifest
- build group manifest
- compute quality metrics

### 9.2 `run`

Purpose:

- select anchors
- run detection
- run propagation
- write fused proposals
- build review queue

### 9.3 `review/update`

Allowed actions:

- `approve`
- `reject`
- `needs_edit`
- `reset`

### 9.4 `export`

Purpose:

- export approved proposals into YOLO dataset format
- never copy derived views
- write `dataset.yaml`

## 10. Report artifacts

Required reports:

- `reports/defect_autolabel_run.json`
- `reports/defect_autolabel_run.md`
- `reports/review_summary.json`
- `reports/export_summary.json`

The markdown report must summarize:

- domain
- asset counts
- lux bucket counts
- anchor count
- propagated proposal count
- review-required count
- class distribution

## 11. Review queue behavior

Every review item must show:

- original image preview
- overlayed boxes
- proposal source
- lux bucket
- quality flags
- class name
- confidence
- reviewer status

The queue must support filters for:

- domain class
- lux bucket
- source mode
- priority
- review status

## 12. Testing requirements

At minimum, add these tests:

- `test_defect_dataset_ingest_parses_group_and_lux`
- `test_defect_quality_assigns_expected_lux_bucket`
- `test_anchor_selection_prefers_best_quality_not_max_lux`
- `test_direct_detection_filters_classes_by_domain`
- `test_propagation_falls_back_when_registration_fails`
- `test_review_queue_marks_dark_and_glare_as_high_priority`
- `test_export_uses_only_original_images`

Critical acceptance criteria:

- paired lux groups are formed correctly from filename or manifest
- anchor selection is deterministic
- export contains only original image paths
- propagation never bypasses registration safety checks
- review queue is created even when propagation fails

## 13. Implementation sequence

### Phase 1

- dataset ingest
- quality metrics
- direct detection on originals plus in-memory normalized view
- review queue
- YOLO export

### Phase 2

- paired-lux grouping
- anchor selection
- registration-based propagation
- merged proposal scoring

### Phase 3

- confidence calibration per lux bucket
- domain-specific false-positive suppression rules
- operator UX cleanup

## 14. Decision summary

This is the shape that should be built next:

- product type: original-only defect auto-labeler
- supported domains: ship defect, metal plate defect
- label type: bbox only
- lighting strategy: infer on original plus temporary normalized views, then export labels on originals only
- lux strategy: choose anchor per group, propagate safely, fall back to direct detection when unsafe
- output: reviewed YOLO detection dataset

If the next step is implementation, build the V2 path beside the current research flow instead of pushing more logic into the already broad V1 services.
