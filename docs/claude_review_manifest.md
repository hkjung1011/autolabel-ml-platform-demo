# Claude Review Manifest

## Goal

This workspace is a research-first defect vision platform for:

- lux-grouped dataset intake and workspace bootstrap
- Retinex restoration, registration, MergeMertens, defect-aware fusion
- evidence benchmarking and training planning
- auto-label bootstrap and review queue
- segmentation bootstrap from bbox labels into coarse masks/polygons
- optional SAM-refined bootstrap using `C:\autolabel-ml-platform-demo\apps\api\mobile_sam.pt`
- segmentation training-plan path through `seg_bootstrap`
- scorecard, paper pack, CSV export
- Windows desktop packaging as an exe

## Current Runtime Snapshot

- Candidate workspace: `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1`
- Current scores: research `100`, field `85`, production `71`
- Current stage: `Segmentation bootstrap and review`
- Current recommended arm: `retinex80`
- Current deploy candidate: `raw160`
- EXE output: `C:\autolabel-ml-platform-demo\apps\api\dist\DefectVisionResearch.exe`

## Entry Points

- `C:\autolabel-ml-platform-demo\apps\api\src\app\main.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\api\routes.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\static\index.html`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\static\brand_mark.svg`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\desktop_entry.py`

## Core Domain Models

- `C:\autolabel-ml-platform-demo\apps\api\src\app\domain\models.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\domain\research_models.py`

## Core Config

- `C:\autolabel-ml-platform-demo\apps\api\src\app\core\config.py`

## Plugin Interfaces And Demo Glue

- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\interfaces.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\demo_plugins.py`

## Retinex Plugins

- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\retinex\config.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\retinex\msrcr.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\retinex\provider.py`

## Registration Plugins

- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\registration\translation_aligner.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\registration\verifier.py`

## Fusion Plugins

- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\fusion\mertens_baseline.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\fusion\frequency_split.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\fusion\defect_prior.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\plugins\fusion\defect_aware.py`

## Services

- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\research.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\repository.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\retinex_runner.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\registration.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\fusion_runner.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\evaluation.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\benchmark.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\training.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\workspace_runner.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\accuracy_audit.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\autolabel.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\mask_bootstrap.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\commercialization.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\review_queue.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\desktop_package.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\desktop_runtime.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\operator_guide.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\reporting.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\program_status.py`
- `C:\autolabel-ml-platform-demo\apps\api\src\app\services\pipeline.py`

## Packaging And Build Files

- `C:\autolabel-ml-platform-demo\apps\api\packaging\defect_vision_research.spec`
- `C:\autolabel-ml-platform-demo\apps\api\scripts\build_exe.ps1`
- `C:\autolabel-ml-platform-demo\apps\api\pyproject.toml`

## Test Files

- `C:\autolabel-ml-platform-demo\apps\api\tests\test_demo_api.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_research_v1.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_retinex_msrcr.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_registration.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_fusion.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_daf.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_evaluation.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_benchmark.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_commercialization.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_training.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_workspace_runner.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_artifact_preview.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_discovery_stage.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_reporting.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_autolabel.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_segmentation_bootstrap.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_packaging.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_desktop_runtime.py`
- `C:\autolabel-ml-platform-demo\apps\api\tests\test_operator_guide.py`

## Top-Level Docs

- `C:\autolabel-ml-platform-demo\README.md`
- `C:\autolabel-ml-platform-demo\docs\architecture.md`
- `C:\autolabel-ml-platform-demo\docs\db-schema.md`
- `C:\autolabel-ml-platform-demo\docs\phase1-aelc-vision.md`
- `C:\autolabel-ml-platform-demo\docs\claude_review_manifest.md`

## Real Generated Artifacts Worth Reviewing

- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\manifests\summary.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\readiness\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\evidence\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\accuracy_audit\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\autolabel\bootstrap_report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\segmentation_bootstrap\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\review_queue\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\review_queue\approved_export_report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\datasets\autolabel\approved_reviewed`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\packaging\plan.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\desktop_runtime\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\operator_guide\report.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\commercialization\source_catalog.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\commercialization\commercial_plan.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\commercialization\source_catalog.md`
- newly staged candidate workspaces created via the protected-source `Stage Copy` flow under `C:\paint_defect_research\candidate_workspaces\*`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\reporting\scorecard.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\reporting\arm_comparison.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\reporting\program_status.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\paper_pack\paper_pack.json`
- `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations\pipeline\report.json`

## Suggested Claude Review Order

1. `README.md`
2. `docs\architecture.md`
3. `src\app\api\routes.py`
4. `src\app\domain\research_models.py`
5. `src\app\services\research.py`
6. `src\app\services\registration.py`
7. `src\app\services\fusion_runner.py`
8. `src\app\plugins\fusion\defect_aware.py`
9. `src\app\services\training.py`
10. `src\app\services\autolabel.py`
11. `src\app\services\mask_bootstrap.py`
12. `src\app\services\review_queue.py`
13. `src\app\services\reporting.py`
14. `src\app\services\program_status.py`
15. `src\app\static\index.html`
16. `src\app\services\desktop_package.py`
17. `src\app\desktop_entry.py`
18. candidate workspace artifacts under `C:\paint_defect_research\candidate_workspaces\lux_candidate_top1\evaluations`

## What Claude Should Focus On

- architecture consistency across intake, pixel methods, evaluation, autolabel, review, reporting, and packaging
- whether the evidence and score layers overclaim beyond current labels
- whether `retinex / registration / mertens / daf` are cleanly separated
- whether auto-label bootstrap and review queue are enough for a detector-first loop
- whether the UI structure is understandable for an operator
- whether exe packaging strategy is reasonable before a future React desktop shell
- whether the paper-pack / scorecard logic is useful for research reporting
