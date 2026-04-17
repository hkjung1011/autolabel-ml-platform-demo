# Architecture Overview

## Product framing

The system is a data recovery ML platform rather than a plain labeling tool.

The first implementation priority is an `Auto Exposure + Lighting Control + Vision` loop.
This loop sits before enhancement and labeling so the system can improve capture conditions,
not only repair bad frames after the fact.

Primary loop:

1. Import asset
2. Analyze scene conditions
3. Build auto exposure and lighting decision
4. Build enhancement policy
5. Generate enhanced variants
6. Auto-label original and enhanced variants
7. Select the best candidate
8. Route risky outputs to review
9. Generate synthetic support data for missing conditions
10. Train and benchmark on condition-specific subsets
11. Feed failures back into review and recovery queues

## Layers

### Experience Layer

- Dashboard
- Dataset browser
- Pixel workbench
- Annotation review queue
- Benchmark dashboard

### Orchestration Layer

- Pipeline service
- KPI aggregation
- Review routing
- Asset and variant lifecycle control

### Pixel Intelligence Layer

- Scene analysis
- Exposure and lighting control planning
- Enhancement planning
- Variant generation
- Quality and artifact scoring

## Phase 1 feature focus

The first release should optimize for capture readiness:

- Detect low-light, backlight, clipping, blur, and noise
- Recommend exposure bias, gain, shutter strategy, and lighting action
- Estimate a vision-ready score before running enhancement
- Use that decision to influence enhancement and downstream labeling

### Label Intelligence Layer

- Auto-label provider abstraction
- Candidate comparison
- Transform safety policy
- Review queue creation

### Synthetic Data Layer

- Gap-aware synthetic recipe generation
- Variant production hooks
- Synthetic label projection
- Synthetic effectiveness tracking

### Training & Benchmark Layer

- Experiment modes: original-only, enhanced-only, mixed, synthetic-mixed
- Subset benchmarks for low-light, backlight, noise-heavy assets
- KPI rollups

### Data/Version Layer

- Assets and variants
- Schema and annotation candidates
- Review decisions
- Benchmark outputs
- Audit log

## Demo implementation choice

The demo backend is intentionally lightweight:

- FastAPI API
- In-memory repository
- Filesystem artifact storage
- Heuristic plugins instead of production CV models

This keeps the module boundaries visible while remaining runnable.

## V1 implementation status

The current V1 adds a functional research bootstrap flow:

- scan dataset root
- discover lux-organized candidates on the external drive when the exact dataset root is not known yet
- stage a bounded candidate subset into the local research workspace before running the heavier phases
- build pair manifest by exposure group
- freeze train/val/test buckets by `group_id` to avoid cross-exposure leakage
- identify `lux160` labeled anchors
- materialize a V1 study set under a local SSD workspace
- emit an experiment plan for Retinex and defect-aware fusion
- execute the first Retinex baseline from the same workspace using stored manifests
- verify transformed variants through a common registration gate before label reuse
- generate a MergeMertens baseline branch from the same manifest set for Phase 4 comparison
- generate a defect-aware fusion branch that keeps low-frequency illumination fusion and boosts high-frequency defect detail
- materialize accepted registered variants and fusion outputs back into frozen train/val/test datasets
- generate an evaluation-readiness report that measures phase completion, dataset-arm readiness, and missing dependencies before full training
- distinguish structure completion from execution readiness so the dashboard does not overstate how close the project is to runnable A/B training
- generate an evidence benchmark report that shows why a branch is better using defect-visibility proxy metrics before full detector training
- separate the current coverage-adjusted recommendation from the raw peak branch so UI decisions are not biased by tiny comparable subsets
- auto-load the latest readiness and evidence reports so the workspace opens with the current project state instead of an empty dashboard
- generate dry-run training and ablation plans from the same workspace so the user can validate arm readiness before installing YOLO dependencies
- orchestrate the full candidate workspace loop with a single API call so newly staged external-drive subsets can be evaluated immediately
- expose a visual pipeline runner in the UI so the user can watch stage progress, inspect logs, and compare intermediate image artifacts while the run is happening
- support live training jobs through an external trainer command so the UI can orchestrate a separate GPU/WSL runtime instead of assuming all ML dependencies live inside the app process
- expose an operational scorecard plus Excel-compatible CSV exports so stakeholders can judge research readiness, field usability, and production gaps from the same workspace
- expose a program-status layer that rolls structure completion, auto-label progress, segmentation readiness, blockers, and module status into one operator-facing dashboard view
- expose an accuracy-audit layer that separates detection-ready workspaces from true segmentation-ready workspaces before over-claiming metrics
- expose an auto-label bootstrap layer that merges reusable labels from raw, registered, and fusion arms into a first detector-training seed dataset
- expose a segmentation-bootstrap layer that converts bbox labels into review-required coarse masks and YOLO segmentation polygons for the next mask-review pass
- expose an optional SAM-refined segmentation-bootstrap path that upgrades bbox prompts into reviewed mask candidates with `mobile_sam.pt`
- expose a `seg_bootstrap` training arm so segmentation-ready datasets can emit `yolo segment train` plans from the same workspace
- expose a review-queue layer so those proposals become explicit human validation work instead of silent filesystem outputs
- promote the review-queue layer into an operator workflow with ownership, notes, review history, and approved-set export for the next retraining pass
- expose a desktop-packaging layer so the same app can evolve from a research tool into a Windows exe with preserved UI and local server behavior
- expose a protected-source-catalog layer so external HDD datasets are treated as immutable sources and all experimentation is staged by copy
- expose a protected-source staging layer so a catalog entry can be copied into a fresh candidate workspace and run through the same pipeline without touching the original source
- expose a commercialization-plan layer so research score, field score, packaging state, and source expansion turn into a pilot roadmap instead of ad hoc notes
- expose commercial status inside the top operator status view so commercialization readiness is tracked alongside research, field, and production scores
- expose a quick-start operator deck with session persistence and desktop-mode UX so the packaged exe behaves like an operator console rather than a raw dev page
- expose desktop runtime diagnostics so an operator can see EXE readiness, missing dependencies, and next actions before running the full workflow
- expose an operator-guide layer so runtime health, program status, commercialization state, and next clicks become a guided workflow instead of a raw dashboard
- expose a paper-pack layer so the same workspace can emit title candidates, abstract drafts, ablation rows, and reproducibility notes for paper writing with existing datasets
