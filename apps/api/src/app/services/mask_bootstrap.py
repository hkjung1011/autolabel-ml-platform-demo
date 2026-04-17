from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.atomic_io import atomic_save_image, atomic_write_text
from app.domain.research_models import (
    EvaluationRunRequest,
    SegmentationBootstrapRequest,
    SegmentationBootstrapResponse,
)
from app.services.evaluation import evaluation_service


class MaskBootstrapService:
    def build_dataset(self, request: SegmentationBootstrapRequest) -> SegmentationBootstrapResponse:
        workspace_path = Path(request.workspace_root)
        source_name, source_root = self._resolve_source_dataset(workspace_path, request.source_dataset_name)
        dataset_root = workspace_path / "datasets" / "segmentation_bootstrap" / "coarse_masks"
        if request.overwrite and dataset_root.exists():
            self._safe_rmtree(dataset_root, workspace_path)

        for relative in [
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
            "masks/train",
            "masks/val",
            "masks/test",
            "meta",
        ]:
            (dataset_root / relative).mkdir(parents=True, exist_ok=True)

        split_counts = {"train": 0, "val": 0, "test": 0}
        class_ids: set[int] = set()
        sample_mask_paths: list[str] = []
        manifest_rows: list[dict[str, str | int | list[str]]] = []
        sam_predictor = self._load_sam_predictor(request) if request.refine_with_sam else None
        sam_device = self._resolve_sam_device(request.sam_device) if sam_predictor is not None else None
        refined_items = 0

        for split in ("train", "val", "test"):
            if split not in request.include_splits:
                continue
            image_dir = source_root / "images" / split
            label_dir = source_root / "labels" / split
            if not image_dir.exists() or not label_dir.exists():
                continue
            for image_path in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                segments = self._convert_bbox_to_segments(
                    image_path=image_path,
                    label_path=label_path,
                    padding_ratio=request.padding_ratio,
                    min_padding_px=request.min_padding_px,
                )
                if not segments:
                    continue
                bootstrap_note = "coarse bootstrap from bbox labels"
                if sam_predictor is not None:
                    refined_segments = self._refine_segments_with_sam(
                        predictor=sam_predictor,
                        image_path=image_path,
                        segments=segments,
                        device=sam_device or "cpu",
                    )
                    if refined_segments is not None:
                        segments = refined_segments
                        refined_items += 1
                        bootstrap_note = "SAM-refined bootstrap from bbox prompts"

                dest_image = dataset_root / "images" / split / image_path.name
                dest_label = dataset_root / "labels" / split / f"{image_path.stem}.txt"
                dest_mask = dataset_root / "masks" / split / f"{image_path.stem}.png"
                shutil.copy2(image_path, dest_image)
                self._write_segmentation_label(dest_label, segments)
                self._write_mask(dest_mask, image_path, segments)

                split_counts[split] = split_counts.get(split, 0) + 1
                for segment in segments:
                    class_ids.add(segment["class_id"])
                if len(sample_mask_paths) < 8:
                    sample_mask_paths.append(str(dest_mask))
                manifest_rows.append(
                    {
                        "split": split,
                        "source_image_path": str(image_path),
                        "source_label_path": str(label_path),
                        "output_image_path": str(dest_image),
                        "output_label_path": str(dest_label),
                        "output_mask_path": str(dest_mask),
                        "segment_count": len(segments),
                        "notes": [
                            bootstrap_note,
                            "review required before segmentation training claims",
                        ],
                    }
                )

        dataset_yaml_path = self._write_dataset_yaml(dataset_root, sorted(class_ids))
        report_root = workspace_path / "evaluations" / "segmentation_bootstrap"
        report_root.mkdir(parents=True, exist_ok=True)
        response = SegmentationBootstrapResponse(
            workspace_root=str(workspace_path),
            source_dataset_name=source_name,
            source_dataset_root=str(source_root),
            dataset_root=str(dataset_root),
            dataset_yaml_path=str(dataset_yaml_path),
            mask_root=str(dataset_root / "masks"),
            total_items=sum(split_counts.values()),
            split_counts=split_counts,
            class_ids=sorted(class_ids),
            sample_mask_paths=sample_mask_paths,
            bootstrap_mode="sam_refined_mask" if refined_items else "coarse_box_mask",
            sam_used=refined_items > 0,
            sam_model=request.sam_model if request.refine_with_sam else None,
            sam_device=sam_device,
            refined_items=refined_items,
            review_required_count=sum(split_counts.values()),
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
            message="Segmentation bootstrap dataset created. Review is required before training claims.",
        )
        self._write_report(response, manifest_rows)
        return response

    def load_report(self, workspace_root: str) -> SegmentationBootstrapResponse:
        report_path = Path(workspace_root) / "evaluations" / "segmentation_bootstrap" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing segmentation bootstrap report: {report_path}")
        return SegmentationBootstrapResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _resolve_source_dataset(self, workspace_path: Path, source_dataset_name: str) -> tuple[str, Path]:
        approved_root = workspace_path / "datasets" / "autolabel" / "approved_reviewed"
        bootstrap_root = workspace_path / "datasets" / "autolabel" / "bootstrap_merged"
        readiness = self._load_readiness(workspace_path)
        if source_dataset_name == "auto":
            if approved_root.exists() and self._count_label_files(approved_root) >= 10:
                return "approved_reviewed", approved_root
            if bootstrap_root.exists():
                return "bootstrap_merged", bootstrap_root
            raw_arm = next((arm for arm in readiness.arms if arm.arm_name == "raw160"), None)
            if raw_arm is None:
                raise FileNotFoundError("Could not resolve a source dataset for segmentation bootstrap.")
            return "raw160", Path(raw_arm.dataset_root)

        if source_dataset_name == "approved_reviewed":
            if not approved_root.exists():
                raise FileNotFoundError(f"Approved reviewed dataset is missing: {approved_root}")
            return source_dataset_name, approved_root
        if source_dataset_name == "bootstrap_merged":
            if not bootstrap_root.exists():
                raise FileNotFoundError(f"Auto-label bootstrap dataset is missing: {bootstrap_root}")
            return source_dataset_name, bootstrap_root

        arm = next((item for item in readiness.arms if item.arm_name == source_dataset_name), None)
        if arm is None:
            raise FileNotFoundError(f"Unknown segmentation bootstrap source dataset: {source_dataset_name}")
        return source_dataset_name, Path(arm.dataset_root)

    def _count_label_files(self, dataset_root: Path) -> int:
        label_root = dataset_root / "labels"
        if not label_root.exists():
            return 0
        return sum(1 for _ in label_root.rglob("*.txt"))

    def _load_sam_predictor(self, request: SegmentationBootstrapRequest):
        if not importlib.util.find_spec("ultralytics"):
            return None
        from ultralytics import SAM  # type: ignore

        return SAM(str(self._resolve_sam_model_path(request.sam_model)))

    def _resolve_sam_model_path(self, sam_model: str) -> Path | str:
        candidate = Path(sam_model)
        if candidate.is_absolute():
            return candidate
        app_root = Path(__file__).resolve().parents[3]
        for path in [app_root / "models" / sam_model, app_root / sam_model]:
            if path.exists():
                return path
        return sam_model

    def _resolve_sam_device(self, sam_device: str) -> str:
        if sam_device != "auto":
            return sam_device
        try:
            import torch  # type: ignore

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_readiness(self, workspace_path: Path):
        try:
            return evaluation_service.load_readiness_report(workspace_path)
        except FileNotFoundError:
            return evaluation_service.build_readiness_report(
                EvaluationRunRequest(workspace_root=str(workspace_path), refresh_report=False)
            )

    def _convert_bbox_to_segments(
        self,
        *,
        image_path: Path,
        label_path: Path,
        padding_ratio: float,
        min_padding_px: int,
    ) -> list[dict[str, int | list[float]]]:
        with Image.open(image_path) as image:
            width, height = image.size
        padding_x = max(min_padding_px, round(width * padding_ratio))
        padding_y = max(min_padding_px, round(height * padding_ratio))
        segments: list[dict[str, int | list[float]]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(value) for value in parts[1:])
            x1 = max(0, int(round((cx - (bw / 2)) * width)) - padding_x)
            y1 = max(0, int(round((cy - (bh / 2)) * height)) - padding_y)
            x2 = min(width - 1, int(round((cx + (bw / 2)) * width)) + padding_x)
            y2 = min(height - 1, int(round((cy + (bh / 2)) * height)) + padding_y)
            polygon = self._polygon_from_box(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height)
            segments.append({"class_id": class_id, "polygon": polygon, "box": [x1, y1, x2, y2]})
        return segments

    def _polygon_from_box(self, *, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> list[float]:
        return [
            round(x1 / width, 6),
            round(y1 / height, 6),
            round(x2 / width, 6),
            round(y1 / height, 6),
            round(x2 / width, 6),
            round(y2 / height, 6),
            round(x1 / width, 6),
            round(y2 / height, 6),
        ]

    def _refine_segments_with_sam(self, *, predictor, image_path: Path, segments: list[dict[str, int | list[float]]], device: str):
        bboxes = [segment["box"] for segment in segments]
        try:
            results = predictor.predict(str(image_path), bboxes=bboxes, device=device, verbose=False)
        except Exception:
            return None
        if not results:
            return None
        result = results[0]
        if result.masks is None or len(result.masks.xy) != len(segments):
            return None

        with Image.open(image_path) as image:
            width, height = image.size

        if hasattr(result.masks.data, "detach"):
            mask_data = result.masks.data.detach().cpu().numpy()
        else:
            mask_data = np.asarray(result.masks.data)

        refined_segments: list[dict[str, int | list[float] | np.ndarray]] = []
        for index, segment in enumerate(segments):
            polygon_xy = result.masks.xy[index]
            if polygon_xy is None or len(polygon_xy) < 3:
                return None
            normalized_polygon: list[float] = []
            for x, y in polygon_xy:
                normalized_polygon.extend([round(float(x) / width, 6), round(float(y) / height, 6)])
            refined_segments.append(
                {
                    "class_id": int(segment["class_id"]),
                    "polygon": normalized_polygon,
                    "box": segment["box"],
                    "mask_array": (mask_data[index] > 0.5).astype(np.uint8),
                }
            )
        return refined_segments

    def _write_segmentation_label(self, label_path: Path, segments: list[dict[str, int | list[float]]]) -> None:
        lines: list[str] = []
        for segment in segments:
            polygon = segment["polygon"]
            points = " ".join(str(value) for value in polygon)
            lines.append(f"{segment['class_id']} {points}")
        atomic_write_text(label_path, "\n".join(lines) + "\n", encoding="utf-8")

    def _write_mask(self, mask_path: Path, image_path: Path, segments: list[dict[str, int | list[float]]]) -> None:
        with Image.open(image_path) as image:
            width, height = image.size
        composed_mask = np.zeros((height, width), dtype=np.uint8)
        for segment in segments:
            class_id = int(segment["class_id"])
            if "mask_array" in segment:
                binary_mask = np.asarray(segment["mask_array"], dtype=np.uint8)
                composed_mask[binary_mask > 0] = min(255, class_id + 1)
                continue
            x1, y1, x2, y2 = segment["box"]
            composed_mask[y1 : y2 + 1, x1 : x2 + 1] = min(255, class_id + 1)
        atomic_save_image(mask_path, Image.fromarray(composed_mask, mode="L"))

    def _write_dataset_yaml(self, dataset_root: Path, class_ids: list[int]) -> Path:
        yaml_path = dataset_root / "meta" / "segmentation_bootstrap.yaml"
        names = [f"class_{class_id}" for class_id in class_ids] or ["class_0"]
        lines = [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            "names:",
        ]
        lines.extend([f"  {index}: {name}" for index, name in enumerate(names)])
        atomic_write_text(yaml_path, "\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def _write_report(self, response: SegmentationBootstrapResponse, manifest_rows: list[dict[str, str | int | list[str]]]) -> None:
        report_path = Path(response.report_json_path)
        atomic_write_text(report_path, json.dumps(response.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = report_path.with_name("items.json")
        atomic_write_text(manifest_path, json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# Segmentation Bootstrap",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Source dataset: `{response.source_dataset_name}`",
            f"- Source root: `{response.source_dataset_root}`",
            f"- Dataset root: `{response.dataset_root}`",
            f"- Mask root: `{response.mask_root}`",
            f"- Total items: **{response.total_items}**",
            f"- Review required: **{response.review_required_count}**",
            f"- Bootstrap mode: **{response.bootstrap_mode}**",
            f"- SAM used: **{response.sam_used}**",
            f"- SAM model: `{response.sam_model or 'n/a'}`",
            f"- SAM device: `{response.sam_device or 'n/a'}`",
            f"- Refined items: **{response.refined_items}**",
            "",
            "## Split Counts",
        ]
        for split, count in response.split_counts.items():
            lines.append(f"- `{split}`: {count}")
        lines.extend([
            "",
            "## Notes",
            "- This dataset is a segmentation bootstrap derived from bbox labels and optionally refined with SAM.",
            "- Masks and polygons must be reviewed before claiming segmentation accuracy.",
            f"- Items manifest: `{manifest_path}`",
        ])
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")

    def _safe_rmtree(self, target: Path, workspace_path: Path) -> None:
        resolved_target = target.resolve()
        resolved_workspace = workspace_path.resolve()
        if resolved_workspace not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_target}")
        shutil.rmtree(resolved_target, ignore_errors=False)


mask_bootstrap_service = MaskBootstrapService()
