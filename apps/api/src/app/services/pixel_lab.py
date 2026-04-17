from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import PixelLabResponse, PixelMethodSummary
from app.services.benchmark import benchmark_service


class PixelLabService:
    def build_report(self, workspace_root: str, target_lux: int = 100) -> PixelLabResponse:
        workspace_path = Path(workspace_root)
        evidence = benchmark_service.load_report(workspace_path)
        dynamic_capture_report = self._load_optional_json(workspace_path / "evaluations" / "dynamic_capture" / "report.json")
        methods = [
            PixelMethodSummary(
                method_name="Retinex MSRCR",
                implemented=True,
                status="ready" if self._count_files(workspace_path / "variants" / "retinex_msrcr") > 0 else "idle",
                best_for="low-light recovery",
                summary="Raises shadow detail and normalizes dark exposures before labeling or training.",
                output_count=self._count_files(workspace_path / "variants" / "retinex_msrcr"),
            ),
            PixelMethodSummary(
                method_name="Forensic WDR",
                implemented=True,
                status="ready" if self._count_named_outputs(workspace_path / "variants" / "forensic_wdr", "fused.png") > 0 else "idle",
                best_for="backlight-safe defect evidence",
                summary="Preserves highlight evidence, lifts shadow detail, and boosts local defect contrast without forcing a natural-looking tone map.",
                output_count=self._count_named_outputs(workspace_path / "variants" / "forensic_wdr", "fused.png"),
            ),
            PixelMethodSummary(
                method_name="Dynamic Capture",
                implemented=True,
                status="ready" if dynamic_capture_report else "available",
                best_for="capture bracket planning",
                summary="Plans multi-exposure capture modes and maps each group to the next pixel branch before labeling or training.",
                output_count=int(dynamic_capture_report.get("total_groups", 0)) if dynamic_capture_report else 0,
            ),
            PixelMethodSummary(
                method_name="MergeMertens",
                implemented=True,
                status="ready" if self._count_named_outputs(workspace_path / "variants" / "mertens_baseline", "fused.png") > 0 else "idle",
                best_for="classical exposure fusion",
                summary="Combines multiple exposures into a balanced image without explicit HDR calibration.",
                output_count=self._count_named_outputs(workspace_path / "variants" / "mertens_baseline", "fused.png"),
            ),
            PixelMethodSummary(
                method_name="Defect-Aware Fusion",
                implemented=True,
                status="ready" if self._count_named_outputs(workspace_path / "variants" / "defect_aware_fusion", "fused.png") > 0 else "idle",
                best_for="small-defect emphasis",
                summary="Keeps low-frequency illumination stable while boosting high-frequency defect evidence.",
                output_count=self._count_named_outputs(workspace_path / "variants" / "defect_aware_fusion", "fused.png"),
            ),
            PixelMethodSummary(
                method_name=f"Target-Lux {target_lux}",
                implemented=True,
                status="ready" if self._count_named_outputs(workspace_path / "variants" / f"target_lux_{target_lux}", "fused.png") > 0 else "idle",
                best_for="mid-exposure synthesis",
                summary="Synthesizes a target illumination level from neighboring lux exposures and optionally improves local contrast.",
                output_count=self._count_named_outputs(workspace_path / "variants" / f"target_lux_{target_lux}", "fused.png"),
            ),
            PixelMethodSummary(
                method_name="Local Relighting",
                implemented=False,
                status="planned",
                best_for="region-specific defect boost",
                summary="Planned branch for local brightness control around defect candidates without blowing out the whole image.",
                output_count=0,
            ),
        ]

        recommended_method = evidence.recommended_arm or "retinex80"
        preview_paths = self._collect_previews(workspace_path, target_lux)
        key_points = [
            f"Recommended branch from current evidence: {recommended_method}.",
            "WDR synthesis should be treated as the first research axis when dark and bright images are both available.",
            "Use Retinex first for globally dark images, then compare against Mertens and DAF on the same frozen split.",
            "Use Forensic WDR when backlight or reflective surfaces hide defects but you still need evidence-preserving local contrast.",
            f"Use Target-Lux {target_lux} when you want a controlled mid-exposure representation instead of raw high/low lux views.",
            "Use mixed_raw_forensic_wdr when you want to train on both the original image and the WDR-synthesized counterpart together.",
            "DAF remains the strongest branch when defect visibility matters more than natural appearance.",
        ]

        report_root = workspace_path / "evaluations" / "pixel_lab"
        report_root.mkdir(parents=True, exist_ok=True)
        response = PixelLabResponse(
            workspace_root=str(workspace_path),
            recommended_method=recommended_method,
            target_lux_ready=self._count_named_outputs(workspace_path / "variants" / f"target_lux_{target_lux}", "fused.png") > 0,
            target_lux_value=target_lux,
            methods=methods,
            key_points=key_points,
            preview_paths=preview_paths,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> PixelLabResponse:
        path = Path(workspace_root) / "evaluations" / "pixel_lab" / "report.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing pixel lab report: {path}")
        return PixelLabResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def _collect_previews(self, workspace_path: Path, target_lux: int) -> list[str]:
        previews: list[str] = []
        for root in [
            workspace_path / "variants" / "retinex_msrcr",
            workspace_path / "variants" / "forensic_wdr",
            workspace_path / "variants" / f"target_lux_{target_lux}",
            workspace_path / "variants" / "mertens_baseline",
            workspace_path / "variants" / "defect_aware_fusion",
        ]:
            if not root.exists():
                continue
            sample = next(root.rglob("*.png"), None)
            if sample:
                previews.append(str(sample))
        return previews[:8]

    def _count_files(self, root: Path) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob("*.png") if path.is_file())

    def _count_named_outputs(self, root: Path, filename: str) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob(filename) if path.is_file())

    def _load_optional_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_report(self, report: PixelLabResponse) -> None:
        atomic_write_text(
            report.report_json_path,
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Pixel Lab",
            "",
            f"- Workspace: `{report.workspace_root}`",
            f"- Recommended method: **{report.recommended_method or 'n/a'}**",
            f"- Target lux ready: **{report.target_lux_ready}**",
            f"- Target lux value: **{report.target_lux_value or 'n/a'}**",
            "",
            "## Key Points",
        ]
        for point in report.key_points:
            lines.append(f"- {point}")
        lines.extend(["", "## Methods"])
        for method in report.methods:
            lines.append(
                f"- `{method.method_name}`: implemented={method.implemented}, status={method.status}, outputs={method.output_count}, best_for={method.best_for}"
            )
            lines.append(f"  - {method.summary}")
        atomic_write_text(report.report_markdown_path, "\n".join(lines) + "\n", encoding="utf-8")


pixel_lab_service = PixelLabService()
