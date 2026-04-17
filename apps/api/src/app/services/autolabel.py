from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import (
    AutoLabelBuildRequest,
    AutoLabelBuildResponse,
    EvaluationRunRequest,
)
from app.services.evaluation import evaluation_service


class AutoLabelService:
    PROPOSAL_MODES = {
        "raw160": "anchor_label_reuse",
        "retinex": "registered_variant_reuse",
        "mertens": "fusion_variant_reuse",
        "daf": "fusion_variant_reuse",
    }
    ANOMALY_CLASS_NAMES = {
        "dark_region_anomaly": "dark_region_anomaly",
        "bright_region_anomaly": "bright_region_anomaly",
    }

    def build_bootstrap_dataset(self, request: AutoLabelBuildRequest) -> AutoLabelBuildResponse:
        workspace_path = Path(request.workspace_root)
        readiness = evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=request.workspace_root,
                include_arms=request.include_arms,
                refresh_report=False,
            )
        )

        dataset_root = workspace_path / "datasets" / "autolabel" / "bootstrap_merged"
        if request.overwrite and dataset_root.exists():
            self._safe_rmtree(dataset_root, workspace_path)

        (dataset_root / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels").mkdir(parents=True, exist_ok=True)
        meta_root = dataset_root / "meta"
        meta_root.mkdir(parents=True, exist_ok=True)

        split_counts = {"train": 0, "val": 0, "test": 0}
        arm_counts: dict[str, int] = {}
        proposal_modes: dict[str, int] = {}
        source_datasets: dict[str, str] = {}
        proposals: list[dict[str, str]] = []
        class_ids: set[int] = {
            class_id
            for arm in readiness.arms
            if arm.arm_name in request.include_arms and arm.ready
            for class_id in arm.class_ids
        }
        anomaly_class_map = (
            self._build_anomaly_class_map(class_ids)
            if (request.include_lighting_anomalies or request.focus_mode == "defect_and_lighting_anomaly")
            else {}
        )
        anomaly_type_counts = {key: 0 for key in self.ANOMALY_CLASS_NAMES}
        anomaly_image_count = 0
        anomaly_box_count = 0

        for arm in readiness.arms:
            if arm.arm_name not in request.include_arms or not arm.ready:
                continue
            arm_root = Path(arm.dataset_root)
            source_datasets[arm.arm_name] = str(arm_root)
            proposal_mode = self.PROPOSAL_MODES.get(arm.arm_name, "label_reuse")
            for class_id in arm.class_ids:
                class_ids.add(class_id)
            for split in ("train", "val", "test"):
                image_dir = arm_root / "images" / split
                label_dir = arm_root / "labels" / split
                if not image_dir.exists() or not label_dir.exists():
                    continue
                for image_path in sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}):
                    label_path = label_dir / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        continue
                    dest_image_dir = dataset_root / "images" / split
                    dest_label_dir = dataset_root / "labels" / split
                    dest_image_dir.mkdir(parents=True, exist_ok=True)
                    dest_label_dir.mkdir(parents=True, exist_ok=True)
                    dest_image = dest_image_dir / f"{arm.arm_name}__{image_path.name}"
                    dest_label = dest_label_dir / f"{arm.arm_name}__{label_path.name}"
                    shutil.copy2(image_path, dest_image)
                    original_label_text = label_path.read_text(encoding="utf-8")
                    label_lines = [line.strip() for line in original_label_text.splitlines() if line.strip()]

                    anomaly_lines: list[str] = []
                    anomaly_types_for_image: list[str] = []
                    if anomaly_class_map:
                        generated_lines, generated_counts = self._build_lighting_anomaly_labels(
                            image_path=image_path,
                            anomaly_class_map=anomaly_class_map,
                            dark_threshold=request.dark_threshold,
                            bright_threshold=request.bright_threshold,
                            min_region_area_ratio=request.min_region_area_ratio,
                        )
                        anomaly_lines.extend(generated_lines)
                        for anomaly_type, count in generated_counts.items():
                            if count:
                                anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + count
                                anomaly_box_count += count
                                anomaly_types_for_image.append(anomaly_type)
                        if anomaly_lines:
                            anomaly_image_count += 1
                            class_ids.update(anomaly_class_map.values())

                    atomic_write_text(
                        dest_label,
                        "\n".join(label_lines + anomaly_lines) + ("\n" if (label_lines or anomaly_lines) else ""),
                        encoding="utf-8",
                    )

                    proposals.append(
                        {
                            "arm_name": arm.arm_name,
                            "split": split,
                            "proposal_mode": proposal_mode,
                            "focus_mode": request.focus_mode,
                            "anomaly_types": anomaly_types_for_image,
                            "anomaly_box_count": len(anomaly_lines),
                            "source_image_path": str(image_path),
                            "source_label_path": str(label_path),
                            "output_image_path": str(dest_image),
                            "output_label_path": str(dest_label),
                        }
                    )
                    split_counts[split] += 1
                    arm_counts[arm.arm_name] = arm_counts.get(arm.arm_name, 0) + 1
                    proposal_modes[proposal_mode] = proposal_modes.get(proposal_mode, 0) + 1

        dataset_yaml_path = self._write_dataset_yaml(dataset_root, sorted(class_ids))
        report_root = workspace_path / "evaluations" / "autolabel"
        report_root.mkdir(parents=True, exist_ok=True)
        class_names = [f"class_{class_id}" for class_id in sorted(class_ids)] or ["class_0"]
        response = AutoLabelBuildResponse(
            workspace_root=str(workspace_path),
            dataset_root=str(dataset_root),
            dataset_yaml_path=str(dataset_yaml_path),
            total_proposals=len(proposals),
            focus_mode=request.focus_mode,
            class_names=class_names,
            split_counts=split_counts,
            arm_counts=arm_counts,
            proposal_modes=proposal_modes,
            anomaly_box_count=anomaly_box_count,
            anomaly_image_count=anomaly_image_count,
            anomaly_type_counts=anomaly_type_counts,
            source_datasets=source_datasets,
            report_json_path=str(report_root / "bootstrap_report.json"),
            report_markdown_path=str(report_root / "bootstrap_report.md"),
            message="Auto-label bootstrap dataset created from reusable labels and optional lighting-anomaly proposals across the selected arms.",
        )
        self._write_report(response, proposals)
        return response

    def load_bootstrap_report(self, workspace_root: str) -> AutoLabelBuildResponse:
        report_path = Path(workspace_root) / "evaluations" / "autolabel" / "bootstrap_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing autolabel bootstrap report: {report_path}")
        return AutoLabelBuildResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _write_dataset_yaml(self, dataset_root: Path, class_ids: list[int]) -> Path:
        yaml_path = dataset_root / "meta" / "autolabel_bootstrap.yaml"
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

    def _build_anomaly_class_map(self, class_ids: set[int]) -> dict[str, int]:
        next_class_id = (max(class_ids) + 1) if class_ids else 1
        class_map: dict[str, int] = {}
        for anomaly_type in self.ANOMALY_CLASS_NAMES:
            class_map[anomaly_type] = next_class_id
            next_class_id += 1
        return class_map

    def _build_lighting_anomaly_labels(
        self,
        *,
        image_path: Path,
        anomaly_class_map: dict[str, int],
        dark_threshold: int,
        bright_threshold: int,
        min_region_area_ratio: float,
    ) -> tuple[list[str], dict[str, int]]:
        image = self._read_grayscale_image(image_path)
        if image is None:
            return [], {key: 0 for key in anomaly_class_map}
        height, width = image.shape[:2]
        min_area = max(24, int(height * width * max(0.001, min_region_area_ratio)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        masks = {
            "dark_region_anomaly": (image <= dark_threshold).astype(np.uint8) * 255,
            "bright_region_anomaly": (image >= bright_threshold).astype(np.uint8) * 255,
        }
        lines: list[str] = []
        counts = {key: 0 for key in anomaly_class_map}

        for anomaly_type, mask in masks.items():
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ranked_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
            for contour in ranked_contours:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                area = box_width * box_height
                if area < min_area:
                    continue
                width_ratio = box_width / float(width)
                height_ratio = box_height / float(height)
                if width_ratio < 0.04 and height_ratio < 0.04:
                    continue
                x_center = (x + (box_width / 2.0)) / float(width)
                y_center = (y + (box_height / 2.0)) / float(height)
                lines.append(
                    f"{anomaly_class_map[anomaly_type]} "
                    f"{x_center:.6f} {y_center:.6f} {width_ratio:.6f} {height_ratio:.6f}"
                )
                counts[anomaly_type] += 1
        return lines, counts

    def _read_grayscale_image(self, image_path: Path) -> np.ndarray | None:
        try:
            buffer = np.fromfile(str(image_path), dtype=np.uint8)
        except OSError:
            return None
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    def _write_report(self, response: AutoLabelBuildResponse, proposals: list[dict[str, str]]) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        proposals_path = Path(response.report_json_path).with_name("proposals.json")
        atomic_write_text(proposals_path, json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Auto-Label Bootstrap",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Dataset root: `{response.dataset_root}`",
            f"- Dataset yaml: `{response.dataset_yaml_path}`",
            f"- Total proposals: **{response.total_proposals}**",
            f"- Focus mode: `{response.focus_mode}`",
            f"- Lighting anomaly boxes: **{response.anomaly_box_count}** across **{response.anomaly_image_count}** images",
            "",
            "## Split Counts",
        ]
        for split, count in response.split_counts.items():
            lines.append(f"- `{split}`: {count}")
        lines.extend(["", "## Arm Counts"])
        for arm, count in response.arm_counts.items():
            lines.append(f"- `{arm}`: {count}")
        lines.extend(["", "## Proposal Modes"])
        for mode, count in response.proposal_modes.items():
            lines.append(f"- `{mode}`: {count}")
        lines.extend(["", "## Anomaly Types"])
        for anomaly_type, count in response.anomaly_type_counts.items():
            lines.append(f"- `{anomaly_type}`: {count}")
        lines.extend(["", "## Source Datasets"])
        for arm, root in response.source_datasets.items():
            lines.append(f"- `{arm}`: `{root}`")
        lines.extend([
            "",
            "## Next Use",
            "- Use this merged dataset as the first detector-training seed set.",
            "- When `defect_and_lighting_anomaly` mode is enabled, review bright/dark region proposals before using them as product labels.",
            "- After the first detector run, replace plain label reuse with detector-driven auto-label proposals plus review.",
            f"- Proposal detail JSON: `{proposals_path}`",
        ])
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")

    def _safe_rmtree(self, target: Path, workspace_path: Path) -> None:
        resolved_target = target.resolve()
        resolved_workspace = workspace_path.resolve()
        if resolved_workspace not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_target}")
        shutil.rmtree(resolved_target, ignore_errors=False)


autolabel_service = AutoLabelService()
