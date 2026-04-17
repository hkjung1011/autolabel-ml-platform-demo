# Future DB Schema

The demo keeps runtime state in memory, but the target PostgreSQL schema should include these core tables.

## Core entities

- `projects`
- `datasets`
- `dataset_versions`
- `assets`
- `asset_variants`
- `scene_conditions`
- `enhancement_profiles`
- `enhancement_runs`
- `quality_metrics`
- `artifact_metrics`
- `label_schemas`
- `annotations`
- `annotation_candidates`
- `annotation_reviews`
- `synthetic_recipes`
- `synthetic_asset_links`
- `jobs`
- `model_registry`
- `training_runs`
- `evaluation_results`
- `subset_benchmarks`
- `audit_logs`

## Critical relationships

- One `asset` has many `asset_variants`
- One `asset_variant` has one `scene_condition`
- One `asset_variant` has many `annotation_candidates`
- One `annotation_candidate` may lead to one `annotation_review`
- One `dataset_version` includes many assets and variants
- One `training_run` produces many `evaluation_results` and `subset_benchmarks`

## Partitioning suggestions

- Store artifacts in object storage and keep metadata in PostgreSQL
- Partition `audit_logs`, `jobs`, and `training_runs` by time when scale increases
- Add a vector index later for asset similarity and active learning retrieval
