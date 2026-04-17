from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    BootstrapV1Request,
    BootstrapV1Response,
    DatasetDiscoveryRequest,
    DatasetDiscoveryResponse,
    ExperimentArm,
    ExperimentPlan,
    LuxDatasetCandidate,
    PairGroup,
    ResearchDatasetSummary,
    StageCandidateRequest,
    StageCandidateResponse,
    WorkspaceArtifactPaths,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAIR_PATTERN = re.compile(r"^(?P<prefix>.+?)_lux(?P<lux>\d+)_(?P<shot>\d+-\d+)$", re.IGNORECASE)
ANCHOR_LUX = "160"
FROZEN_SPLIT_BUCKETS = (
    ("train", 70),
    ("val", 15),
    ("test", 15),
)
UNMATCHED_SAMPLE_LIMIT = 20
KNOWN_SPLITS = {"train", "val", "test"}


@dataclass
class ResearchWorkspaceState:
    response: BootstrapV1Response | None = None


class ResearchWorkspaceService:
    def __init__(self) -> None:
        self.state = ResearchWorkspaceState()

    def latest(self) -> BootstrapV1Response | None:
        return self.state.response

    def bootstrap_v1(self, request: BootstrapV1Request) -> BootstrapV1Response:
        dataset_root = request.dataset_dir()
        scan = self._scan_dataset(dataset_root)
        images_root = scan["images_root"]

        workspace_root = request.workspace_dir()
        self._ensure_workspace_dirs(workspace_root)
        pair_groups = scan["pair_groups"]
        labeled_groups = scan["labeled_groups"]

        artifacts = self._write_workspace_artifacts(
            workspace_root=workspace_root,
            pair_groups=pair_groups,
            labeled_groups=labeled_groups,
            weights_path=request.weights_path,
            dataset_path=str(dataset_root),
        )

        if request.materialize_workspace:
            self._materialize_v1_study_set(
                workspace_root=workspace_root,
                images_root=images_root,
                labeled_groups=labeled_groups,
            )

        experiment_plan = self._build_experiment_plan(weights_path=request.weights_path)
        self._write_json(Path(artifacts.experiment_plan_path), experiment_plan.model_dump())

        summary = ResearchDatasetSummary(
            dataset_path=str(dataset_root),
            weights_path=request.weights_path,
            total_images=scan["total_images"],
            total_groups=len(pair_groups),
            total_size_gb=round(scan["total_size_bytes"] / (1024**3), 3),
            lux_counts=dict(sorted(scan["lux_counts"].items(), key=lambda item: int(item[0]))),
            combo_counts=dict(sorted(scan["combo_counts"].items())),
            labeled_anchor_count=scan["labeled_anchor_count"],
            labeled_with_40=scan["labeled_with_40"],
            labeled_with_80=scan["labeled_with_80"],
            labeled_with_both=scan["labeled_with_both"],
            triple_group_count=scan["triple_group_count"],
            experiment_ready_count=scan["experiment_ready_count"],
            baseline_ready_count=scan["baseline_ready_count"],
            frozen_split_counts=dict(sorted(scan["frozen_split_counts"].items())),
            labeled_split_counts=dict(sorted(scan["labeled_split_counts"].items())),
            unmatched_image_count=scan["unmatched_image_count"],
            unmatched_image_samples=scan["unmatched_images"],
            sample_groups=pair_groups[:12],
            artifact_paths=artifacts,
        )
        self._write_json(Path(artifacts.summary_path), summary.model_dump())

        response = BootstrapV1Response(
            summary=summary,
            experiment_plan=experiment_plan,
            pair_groups=pair_groups[:80],
            message="V1 workspace is ready. Dataset scan, manifests, and study-set materialization completed.",
        )
        self.state.response = response
        return response

    def discover_candidates(self, request: DatasetDiscoveryRequest) -> DatasetDiscoveryResponse:
        scan_root = Path(request.scan_root)
        if not scan_root.exists():
            raise FileNotFoundError(f"Scan root does not exist: {scan_root}")

        dataset_roots: set[Path] = set()
        for image_path in scan_root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not PAIR_PATTERN.match(image_path.stem):
                continue
            dataset_root = self._infer_dataset_root(image_path)
            if dataset_root is not None:
                dataset_roots.add(dataset_root)

        candidates: list[LuxDatasetCandidate] = []
        for dataset_root in sorted(dataset_roots):
            try:
                scan = self._scan_dataset(dataset_root)
            except (FileNotFoundError, ValueError):
                continue
            if scan["total_images"] < request.min_images:
                continue
            lux_levels = len(scan["lux_counts"])
            notes: list[str] = []
            if scan["labeled_anchor_count"] == 0:
                notes.append("No anchor labels found yet.")
            if scan["experiment_ready_count"] == 0:
                notes.append("No labeled companion groups ready for comparison.")
            score = float(
                scan["group_count"]
                + (scan["labeled_anchor_count"] * 2)
                + (scan["experiment_ready_count"] * 2)
                + (lux_levels * 5)
            )
            sample_image_path = None
            if scan["pair_groups"]:
                first_group = scan["pair_groups"][0]
                sample_image_path = first_group.anchor_image_path or next(iter(first_group.exposures.values()), None)
            dataset_name = dataset_root.name
            if dataset_name.lower() in KNOWN_SPLITS and dataset_root.parent != dataset_root:
                dataset_name = f"{dataset_root.parent.name}_{dataset_name}"
            candidates.append(
                LuxDatasetCandidate(
                    dataset_root=str(dataset_root),
                    dataset_name=dataset_name,
                    images_root=str(scan["images_root"]),
                    labels_root=str(scan["labels_root"]) if scan["labels_root"].exists() else None,
                    image_count=scan["total_images"],
                    group_count=scan["group_count"],
                    labeled_anchor_count=scan["labeled_anchor_count"],
                    lux_counts=dict(sorted(scan["lux_counts"].items(), key=lambda item: int(item[0]))),
                    sample_image_path=sample_image_path,
                    score=round(score, 2),
                    notes=notes,
                )
            )

        candidates.sort(key=lambda item: (item.score, item.image_count), reverse=True)
        limited = candidates[: max(1, request.limit)]
        return DatasetDiscoveryResponse(
            scan_root=str(scan_root),
            candidates=limited,
            message=f"Discovered {len(limited)} lux-organized dataset candidates under {scan_root}.",
        )

    def stage_candidate(self, request: StageCandidateRequest) -> StageCandidateResponse:
        source_dataset_root = Path(request.source_dataset_root)
        scan = self._scan_dataset(source_dataset_root)
        groups = scan["labeled_groups"] if request.prefer_labeled and scan["labeled_groups"] else scan["pair_groups"]
        selected_groups = self._select_stage_groups(groups, request.max_groups)
        if not selected_groups:
            raise ValueError("No candidate groups were selected for staging.")

        staged_name = self._safe_name(request.staged_name or source_dataset_root.name)
        workspace_root = Path(request.workspace_root)
        staged_dataset_root = workspace_root / "imports" / staged_name / "dataset"
        self._prepare_empty_dataset_root(staged_dataset_root)

        copied_images = 0
        copied_labels = 0
        selected_group_ids: list[str] = []
        for group in selected_groups:
            frozen_split = group.frozen_split or group.split
            selected_group_ids.append(group.key)
            for lux, image_path_str in group.exposures.items():
                image_path = Path(image_path_str)
                destination = staged_dataset_root / "images" / frozen_split / image_path.name
                self._copy_file(image_path, destination)
                copied_images += 1
                if lux == ANCHOR_LUX and group.anchor_label_path:
                    label_path = Path(group.anchor_label_path)
                    label_destination = staged_dataset_root / "labels" / frozen_split / f"{image_path.stem}.txt"
                    self._copy_file(label_path, label_destination)
                    copied_labels += 1

        staged_workspace_root = None
        bootstrap_message = None
        if request.bootstrap_after_stage:
            staged_workspace_root = str(workspace_root / "candidate_workspaces" / staged_name)
            bootstrap_response = self.bootstrap_v1(
                BootstrapV1Request(
                    dataset_path=str(staged_dataset_root),
                    workspace_root=staged_workspace_root,
                    materialize_workspace=True,
                )
            )
            bootstrap_message = bootstrap_response.message

        return StageCandidateResponse(
            source_dataset_root=str(source_dataset_root),
            staged_dataset_root=str(staged_dataset_root),
            staged_workspace_root=staged_workspace_root,
            copied_images=copied_images,
            copied_labels=copied_labels,
            selected_group_count=len(selected_groups),
            selected_group_ids=selected_group_ids,
            bootstrap_message=bootstrap_message,
        )

    def _build_experiment_plan(self, weights_path: str | None) -> ExperimentPlan:
        notes = [
            "Phase 0 freeze: all exposure variants of the same group stay in the same train/val/test bucket.",
            "Baseline first: lux160 anchors remain the reference train/eval set.",
            "Every transformed branch must pass the same registration gate before label reuse.",
            "Defect-aware fusion branch should compare against raw160, raw80, Retinex, and MergeMertens baseline.",
        ]
        if weights_path:
            notes.append(f"YOLO prior candidate: {weights_path}")
        return ExperimentPlan(
            title="Paint Defect Research V1",
            phases=[
                "Phase 0: freeze split manifest by group_id",
                "Phase 1: scan dataset and materialize study set",
                "Phase 2: Retinex baseline",
                "Phase 3: registration gate",
                "Phase 4: MergeMertens baseline",
                "Phase 5: defect-aware weighted fusion",
                "Phase 6: A/B detection evaluation",
            ],
            arms=[
                ExperimentArm(
                    name="G0_raw160",
                    stage="baseline",
                    description="Use labeled lux160 anchors without exposure recovery.",
                    inputs=["lux160 anchor"],
                    expected_output="reference mAP and defect recall",
                ),
                ExperimentArm(
                    name="G2_retinex80",
                    stage="retinex",
                    description="Restore lux80 toward lux160 and reuse anchor labels after alignment check.",
                    inputs=["lux80 companion", "Retinex model"],
                    expected_output="restored low-light set for detector comparison",
                ),
                ExperimentArm(
                    name="P0_merge_mertens",
                    stage="fusion",
                    description="Pure exposure fusion baseline using available 40/80/160 companions.",
                    inputs=["lux40", "lux80", "lux160"],
                    expected_output="baseline fusion images",
                ),
                ExperimentArm(
                    name="P2_yolo_prior_fusion",
                    stage="fusion",
                    description="Defect-aware fusion with YOLO prior weighting inside defect-sensitive regions.",
                    inputs=["MergeMertens weights", "YOLO prior map", "high-frequency detail map"],
                    expected_output="paper-track fusion branch",
                ),
            ],
            notes=notes,
        )

    def _scan_dataset(self, dataset_root: Path) -> dict:
        images_root = dataset_root / "images"
        labels_root = dataset_root / "labels"
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
        if not images_root.exists():
            raise FileNotFoundError(f"Missing images directory: {images_root}")
        if not labels_root.exists():
            raise FileNotFoundError(f"Missing labels directory: {labels_root}")

        grouped: dict[str, PairGroup] = {}
        lux_counts: dict[str, int] = {}
        total_size_bytes = 0
        total_images = 0
        unmatched_images: list[str] = []
        unmatched_image_count = 0

        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = PAIR_PATTERN.match(image_path.stem)
            if not match:
                unmatched_image_count += 1
                if len(unmatched_images) < UNMATCHED_SAMPLE_LIMIT:
                    unmatched_images.append(str(image_path))
                continue
            relative = image_path.relative_to(images_root)
            split = self._resolve_split_name(relative, dataset_root.name)
            prefix = match.group("prefix")
            lux = match.group("lux")
            shot = match.group("shot")
            key = f"{split}|{prefix}|{shot}"

            if key not in grouped:
                grouped[key] = PairGroup(
                    key=key,
                    split=split,
                    prefix=prefix,
                    shot_id=shot,
                )
            grouped[key].exposures[lux] = str(image_path)

            lux_counts[lux] = lux_counts.get(lux, 0) + 1
            total_size_bytes += image_path.stat().st_size
            total_images += 1

        pair_groups = sorted(grouped.values(), key=lambda item: (item.split, item.prefix, item.shot_id))
        combo_counts: dict[str, int] = {}
        frozen_split_counts: dict[str, int] = {}
        labeled_split_counts: dict[str, int] = {}
        labeled_anchor_count = 0
        labeled_with_40 = 0
        labeled_with_80 = 0
        labeled_with_both = 0
        triple_group_count = 0
        baseline_ready_count = 0
        experiment_ready_count = 0
        labeled_groups: list[PairGroup] = []

        for group in pair_groups:
            ordered_luxes = sorted(group.exposures.keys(), key=lambda value: int(value))
            combo_key = ",".join(ordered_luxes)
            combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
            group.companion_luxes = [lux for lux in ordered_luxes if lux != ANCHOR_LUX]

            if {"40", "80", ANCHOR_LUX}.issubset(group.exposures.keys()):
                triple_group_count += 1

            anchor_image = group.exposures.get(ANCHOR_LUX)
            group.anchor_image_path = anchor_image
            if anchor_image is None:
                continue
            anchor_path = Path(anchor_image)
            anchor_rel = anchor_path.relative_to(images_root)
            label_path = labels_root / anchor_rel.parent / f"{anchor_path.stem}.txt"
            if label_path.exists():
                group.anchor_label_path = str(label_path)
                group.label_reusable = True
                group.label_line_count = self._count_non_empty_lines(label_path)
                labeled_anchor_count += 1
                baseline_ready_count += 1
                if "40" in group.exposures:
                    labeled_with_40 += 1
                if "80" in group.exposures:
                    labeled_with_80 += 1
                if {"40", "80"}.issubset(group.exposures.keys()):
                    labeled_with_both += 1
                if group.companion_luxes:
                    experiment_ready_count += 1
                labeled_groups.append(group)

        self._assign_frozen_splits(pair_groups)

        for group in pair_groups:
            frozen_split_counts[group.frozen_split] = frozen_split_counts.get(group.frozen_split, 0) + 1
            if group.anchor_label_path:
                labeled_split_counts[group.frozen_split] = labeled_split_counts.get(group.frozen_split, 0) + 1

        return {
            "dataset_root": dataset_root,
            "images_root": images_root,
            "labels_root": labels_root,
            "pair_groups": pair_groups,
            "labeled_groups": labeled_groups,
            "lux_counts": lux_counts,
            "combo_counts": combo_counts,
            "total_size_bytes": total_size_bytes,
            "total_images": total_images,
            "group_count": len(pair_groups),
            "labeled_anchor_count": labeled_anchor_count,
            "labeled_with_40": labeled_with_40,
            "labeled_with_80": labeled_with_80,
            "labeled_with_both": labeled_with_both,
            "triple_group_count": triple_group_count,
            "experiment_ready_count": experiment_ready_count,
            "baseline_ready_count": baseline_ready_count,
            "frozen_split_counts": frozen_split_counts,
            "labeled_split_counts": labeled_split_counts,
            "unmatched_image_count": unmatched_image_count,
            "unmatched_images": unmatched_images,
        }

    def _infer_dataset_root(self, image_path: Path) -> Path | None:
        current = image_path.parent
        while current != current.parent:
            if current.name.lower() == "images":
                return current.parent
            current = current.parent
        return None

    def _select_stage_groups(self, groups: list[PairGroup], max_groups: int) -> list[PairGroup]:
        if len(groups) <= max_groups:
            return groups
        buckets = {"train": [], "val": [], "test": []}
        for group in groups:
            buckets[group.frozen_split or group.split].append(group)

        targets = {
            "train": max(1, round(max_groups * 0.7)),
            "val": max(1, round(max_groups * 0.15)),
            "test": max(1, round(max_groups * 0.15)),
        }
        selected: list[PairGroup] = []
        for split_name in ("train", "val", "test"):
            selected.extend(buckets[split_name][: targets[split_name]])

        if len(selected) < max_groups:
            selected_keys = {group.key for group in selected}
            for group in groups:
                if group.key in selected_keys:
                    continue
                selected.append(group)
                selected_keys.add(group.key)
                if len(selected) >= max_groups:
                    break
        return selected[:max_groups]

    def _prepare_empty_dataset_root(self, dataset_root: Path) -> None:
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        for relative in [
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
        ]:
            (dataset_root / relative).mkdir(parents=True, exist_ok=True)

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "candidate"

    def _resolve_split_name(self, relative_path: Path, dataset_root_name: str) -> str:
        if relative_path.parts and relative_path.parts[0].lower() in KNOWN_SPLITS:
            return relative_path.parts[0].lower()
        root_name = dataset_root_name.lower()
        if root_name in KNOWN_SPLITS:
            return root_name
        return "train"

    def _write_workspace_artifacts(
        self,
        workspace_root: Path,
        pair_groups: list[PairGroup],
        labeled_groups: list[PairGroup],
        weights_path: str | None,
        dataset_path: str,
    ) -> WorkspaceArtifactPaths:
        manifests_dir = workspace_root / "manifests"
        pair_manifest_path = manifests_dir / "pair_manifest.json"
        labeled_manifest_path = manifests_dir / "labeled_pair_manifest.json"
        split_manifest_path = manifests_dir / "split_manifest.json"
        experiment_plan_path = manifests_dir / "experiment_plan.json"
        summary_path = manifests_dir / "summary.json"
        registration_report_dir = workspace_root / "registration_reports"

        self._write_json(
            pair_manifest_path,
            {
                "dataset_path": dataset_path,
                "weights_path": weights_path,
                "anchor_lux": ANCHOR_LUX,
                "groups": [group.model_dump() for group in pair_groups],
            },
        )
        self._write_json(
            labeled_manifest_path,
            {
                "dataset_path": dataset_path,
                "weights_path": weights_path,
                "anchor_lux": ANCHOR_LUX,
                "groups": [group.model_dump() for group in labeled_groups],
            },
        )
        self._write_json(
            split_manifest_path,
            {
                "dataset_path": dataset_path,
                "weights_path": weights_path,
                "anchor_lux": ANCHOR_LUX,
                "split_policy": {name: ratio for name, ratio in FROZEN_SPLIT_BUCKETS},
                "groups": [
                    {
                        "key": group.key,
                        "original_split": group.split,
                        "frozen_split": group.frozen_split,
                        "label_reusable": group.label_reusable,
                        "exposure_luxes": sorted(group.exposures.keys(), key=int),
                    }
                    for group in pair_groups
                ],
            },
        )
        return WorkspaceArtifactPaths(
            workspace_root=str(workspace_root),
            pair_manifest_path=str(pair_manifest_path),
            labeled_manifest_path=str(labeled_manifest_path),
            split_manifest_path=str(split_manifest_path),
            experiment_plan_path=str(experiment_plan_path),
            summary_path=str(summary_path),
            registration_report_dir=str(registration_report_dir),
        )

    def _materialize_v1_study_set(
        self,
        workspace_root: Path,
        images_root: Path,
        labeled_groups: list[PairGroup],
    ) -> None:
        baseline_images = workspace_root / "datasets" / "yolo_baseline" / "images"
        baseline_labels = workspace_root / "datasets" / "yolo_baseline" / "labels"
        reuse_images = workspace_root / "datasets" / "yolo_cross_lux_reuse" / "images"
        reuse_labels = workspace_root / "datasets" / "yolo_cross_lux_reuse" / "labels"
        raw_snapshot = workspace_root / "raw_snapshot" / "v1_study"

        for group in labeled_groups:
            if not group.anchor_image_path or not group.anchor_label_path:
                continue
            anchor_source = Path(group.anchor_image_path)
            anchor_label = Path(group.anchor_label_path)
            anchor_rel = anchor_source.relative_to(images_root)
            frozen_split = group.frozen_split or group.split
            anchor_sub_rel = self._relative_inside_split(anchor_rel)

            self._copy_file(anchor_source, baseline_images / frozen_split / anchor_sub_rel)
            self._copy_file(anchor_label, baseline_labels / frozen_split / anchor_sub_rel.parent / f"{anchor_source.stem}.txt")
            self._copy_file(anchor_source, raw_snapshot / "images" / anchor_rel)
            self._copy_file(anchor_label, raw_snapshot / "labels" / anchor_rel.parent / f"{anchor_source.stem}.txt")

            for lux in group.companion_luxes:
                companion_path_str = group.exposures.get(lux)
                if companion_path_str is None:
                    continue
                companion_source = Path(companion_path_str)
                companion_rel = companion_source.relative_to(images_root)
                companion_sub_rel = self._relative_inside_split(companion_rel)
                self._copy_file(companion_source, reuse_images / frozen_split / companion_sub_rel)
                self._copy_file(
                    anchor_label,
                    reuse_labels / frozen_split / companion_sub_rel.parent / f"{companion_source.stem}.txt",
                )
                self._copy_file(companion_source, raw_snapshot / "images" / companion_rel)
                self._copy_file(anchor_label, raw_snapshot / "labels" / companion_rel.parent / f"{companion_source.stem}.txt")

    def _ensure_workspace_dirs(self, workspace_root: Path) -> None:
        for relative in [
            "manifests",
            "raw_snapshot",
            "registered",
            "registration_reports",
            "retinex_outputs",
            "fusion_outputs",
            "variants/retinex_msrcr",
            "datasets/yolo_baseline/images/train",
            "datasets/yolo_baseline/images/val",
            "datasets/yolo_baseline/images/test",
            "datasets/yolo_baseline/labels/train",
            "datasets/yolo_baseline/labels/val",
            "datasets/yolo_baseline/labels/test",
            "datasets/yolo_cross_lux_reuse/images/train",
            "datasets/yolo_cross_lux_reuse/images/val",
            "datasets/yolo_cross_lux_reuse/images/test",
            "datasets/yolo_cross_lux_reuse/labels/train",
            "datasets/yolo_cross_lux_reuse/labels/val",
            "datasets/yolo_cross_lux_reuse/labels/test",
            "datasets/yolo_retinex",
            "datasets/yolo_fusion",
            "experiments",
            "reports",
        ]:
            (workspace_root / relative).mkdir(parents=True, exist_ok=True)

    def _copy_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _count_non_empty_lines(self, file_path: Path) -> int:
        return sum(1 for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _freeze_split(self, key: str) -> str:
        bucket_value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
        cursor = 0
        for name, ratio in FROZEN_SPLIT_BUCKETS:
            cursor += ratio
            if bucket_value < cursor:
                return name
        return FROZEN_SPLIT_BUCKETS[-1][0]

    def _assign_frozen_splits(self, pair_groups: list[PairGroup]) -> None:
        labeled_groups = [group for group in pair_groups if group.anchor_label_path]
        unlabeled_groups = [group for group in pair_groups if not group.anchor_label_path]
        self._assign_stratified_splits(labeled_groups, keep_eval_minimum=True)
        self._assign_stratified_splits(unlabeled_groups, keep_eval_minimum=False)

    def _assign_stratified_splits(self, groups: list[PairGroup], *, keep_eval_minimum: bool) -> None:
        if not groups:
            return
        ordered = sorted(groups, key=lambda item: item.key)
        targets = self._target_split_counts(len(ordered), keep_eval_minimum=keep_eval_minimum)
        index = 0
        for split_name in ("train", "val", "test"):
            for _ in range(targets[split_name]):
                ordered[index].frozen_split = split_name
                index += 1

    def _target_split_counts(self, total: int, *, keep_eval_minimum: bool) -> dict[str, int]:
        if total <= 0:
            return {"train": 0, "val": 0, "test": 0}
        if total == 1:
            return {"train": 1, "val": 0, "test": 0}
        if total == 2:
            return {"train": 1, "val": 0, "test": 1}

        train = round(total * 0.7)
        val = round(total * 0.15)
        test = total - train - val

        if keep_eval_minimum:
            if val == 0:
                val = 1
            if test == 0:
                test = 1
            while train + val + test > total:
                if train > 1:
                    train -= 1
                elif test > 1:
                    test -= 1
                else:
                    val -= 1
            while train + val + test < total:
                train += 1
        else:
            while train + val + test > total:
                if train >= val and train >= test and train > 1:
                    train -= 1
                elif val > test and val > 0:
                    val -= 1
                elif test > 0:
                    test -= 1
                else:
                    train -= 1
            while train + val + test < total:
                train += 1

        return {"train": train, "val": val, "test": test}

    def _relative_inside_split(self, relative_path: Path) -> Path:
        return Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path(relative_path.name)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


research_workspace_service = ResearchWorkspaceService()
