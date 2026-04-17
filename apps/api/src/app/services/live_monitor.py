from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import LiveMonitorArtifact, LiveMonitorResponse


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TRACKED_SUFFIXES = IMAGE_SUFFIXES | {".json", ".md", ".csv", ".txt", ".yaml", ".yml"}


class LiveMonitorService:
    def build_report(self, workspace_root: str) -> LiveMonitorResponse:
        return self.snapshot(workspace_root, persist=True)

    def load_report(self, workspace_root: str) -> LiveMonitorResponse:
        return self.snapshot(workspace_root, persist=False)

    def snapshot(self, workspace_root: str, *, persist: bool) -> LiveMonitorResponse:
        workspace_path = Path(workspace_root)
        summary = self._load_json(workspace_path / "manifests" / "summary.json")
        registration = self._load_json(workspace_path / "registration_reports" / "retinex_msrcr.json")
        autolabel = self._load_json(workspace_path / "evaluations" / "autolabel" / "bootstrap_report.json")
        segmentation = self._load_json(workspace_path / "evaluations" / "segmentation_bootstrap" / "report.json")
        review_queue = self._load_json(workspace_path / "evaluations" / "review_queue" / "report.json")

        recent_artifacts = self._collect_recent_artifacts(
            workspace_path,
            [
                workspace_path / "variants",
                workspace_path / "evaluations",
                workspace_path / "datasets" / "segmentation_bootstrap" / "coarse_masks" / "masks",
                workspace_path / "datasets" / "autolabel",
            ],
        )
        latest_activity_at = recent_artifacts[0].modified_at if recent_artifacts else None
        preview_paths = [artifact.path for artifact in recent_artifacts if Path(artifact.path).suffix.lower() in IMAGE_SUFFIXES][:8]

        report_root = workspace_path / "evaluations" / "live_monitor"
        report_root.mkdir(parents=True, exist_ok=True)
        response = LiveMonitorResponse(
            workspace_root=str(workspace_path),
            source_dataset_path=summary.get("dataset_path"),
            source_image_count=int(summary.get("total_images", 0)),
            staged_image_count=self._count_files(workspace_path / "datasets" / "yolo_baseline" / "images", IMAGE_SUFFIXES),
            retinex_output_count=self._count_files(workspace_path / "variants" / "retinex_msrcr", IMAGE_SUFFIXES),
            forensic_wdr_output_count=self._count_named_outputs(workspace_path / "variants" / "forensic_wdr", "fused.png"),
            registered_variant_count=int(registration.get("accepted_count", 0)) if registration else self._count_files(workspace_path / "datasets" / "registered_variants", IMAGE_SUFFIXES),
            mertens_output_count=self._count_named_outputs(workspace_path / "variants" / "mertens_baseline", "fused.png"),
            daf_output_count=self._count_named_outputs(workspace_path / "variants" / "defect_aware_fusion", "fused.png"),
            autolabel_proposal_count=int(autolabel.get("total_proposals", 0)) if autolabel else 0,
            autolabel_anomaly_box_count=int(autolabel.get("anomaly_box_count", 0)) if autolabel else 0,
            autolabel_focus_mode=autolabel.get("focus_mode") if autolabel else None,
            segmentation_mask_count=int(segmentation.get("total_items", 0)) if segmentation else 0,
            segmentation_refined_items=int(segmentation.get("refined_items", 0)) if segmentation else 0,
            reviewed_count=int(review_queue.get("reviewed_count", 0)) if review_queue else 0,
            approved_count=int((review_queue.get("status_counts") or {}).get("approved", 0)) if review_queue else 0,
            latest_activity_at=latest_activity_at,
            preview_paths=preview_paths,
            recent_artifacts=recent_artifacts,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        if persist:
            self._write_report(response)
        return response

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _count_files(self, root: Path, suffixes: set[str]) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)

    def _count_named_outputs(self, root: Path, filename: str) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob(filename) if path.is_file())

    def _collect_recent_artifacts(self, workspace_path: Path, roots: list[Path]) -> list[LiveMonitorArtifact]:
        artifacts: list[LiveMonitorArtifact] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in TRACKED_SUFFIXES:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = path.relative_to(workspace_path)
                artifacts.append(
                    LiveMonitorArtifact(
                        label=path.name,
                        path=str(path),
                        category=self._classify_artifact(relative),
                        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        size_bytes=stat.st_size,
                    )
                )
        artifacts.sort(key=lambda item: item.modified_at, reverse=True)
        return artifacts[:12]

    def _classify_artifact(self, relative: Path) -> str:
        text = str(relative).lower()
        if "segmentation_bootstrap" in text:
            return "segmentation"
        if "autolabel" in text or "review_queue" in text:
            return "autolabel"
        if "retinex" in text:
            return "retinex"
        if "forensic_wdr" in text:
            return "forensic_wdr"
        if "mertens" in text:
            return "mertens"
        if "defect_aware" in text or "daf" in text:
            return "daf"
        if "training" in text:
            return "training"
        if "pipeline" in text:
            return "pipeline"
        return "artifact"

    def _write_report(self, report: LiveMonitorResponse) -> None:
        atomic_write_text(
            report.report_json_path,
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Live Monitor",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Source dataset: `{report.source_dataset_path or 'n/a'}`",
            f"- Source images: **{report.source_image_count}**",
            f"- Staged images: **{report.staged_image_count}**",
            f"- Retinex outputs: **{report.retinex_output_count}**",
            f"- Forensic WDR outputs: **{report.forensic_wdr_output_count}**",
            f"- Registered variants: **{report.registered_variant_count}**",
            f"- Mertens outputs: **{report.mertens_output_count}**",
            f"- DAF outputs: **{report.daf_output_count}**",
            f"- Auto-label proposals: **{report.autolabel_proposal_count}**",
            f"- Auto-label anomaly boxes: **{report.autolabel_anomaly_box_count}**",
            f"- Segmentation masks: **{report.segmentation_mask_count}**",
            f"- Segmentation refined items: **{report.segmentation_refined_items}**",
            f"- Reviewed: **{report.reviewed_count}** / Approved: **{report.approved_count}**",
            f"- Latest activity: **{report.latest_activity_at or 'n/a'}**",
            "",
            "## Recent Artifacts",
        ]
        for artifact in report.recent_artifacts:
            lines.append(f"- `{artifact.category}` `{artifact.label}` @ {artifact.modified_at}")
            lines.append(f"  - path: `{artifact.path}`")
        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")


live_monitor_service = LiveMonitorService()
