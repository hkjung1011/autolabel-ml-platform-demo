from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import DesktopRuntimeCheckItem, DesktopRuntimeCheckResponse
from app.services.desktop_package import desktop_package_service


class DesktopRuntimeService:
    def build_report(self, workspace_root: str) -> DesktopRuntimeCheckResponse:
        workspace_path = Path(workspace_root)
        package_plan = desktop_package_service.build_plan(str(workspace_path))

        uv_available = shutil.which("uv") is not None
        pyinstaller_ready = package_plan.pyinstaller_ready
        ultralytics_available = self._has_module("ultralytics")
        opencv_available = self._has_module("cv2")
        torch_available = self._has_module("torch")
        cuda_available, gpu_summary = self._gpu_snapshot(torch_available)

        report_root = workspace_path / "evaluations" / "desktop_runtime"
        report_root.mkdir(parents=True, exist_ok=True)

        checks = [
            DesktopRuntimeCheckItem(
                check_name="Workspace Root",
                status="ok" if workspace_path.exists() else "warn",
                summary="Operator workspace path is available for report loading and artifact writes." if workspace_path.exists() else "Workspace root does not exist yet.",
                detail=str(workspace_path),
                action=None if workspace_path.exists() else "Create or bootstrap the workspace before using operator mode.",
            ),
            DesktopRuntimeCheckItem(
                check_name="UV Runtime",
                status="ok" if uv_available else "warn",
                summary="uv command is available for local launch and dependency sync." if uv_available else "uv command is not available in PATH.",
                detail=shutil.which("uv") or "not found",
                action=None if uv_available else "Install uv or use the packaged EXE for operator use.",
            ),
            DesktopRuntimeCheckItem(
                check_name="PyInstaller",
                status="ok" if pyinstaller_ready else "warn",
                summary="PyInstaller is ready for EXE packaging." if pyinstaller_ready else "PyInstaller is missing, so EXE rebuild is blocked.",
                detail=package_plan.spec_path,
                action=None if pyinstaller_ready else "Install PyInstaller in the packaging environment.",
            ),
            DesktopRuntimeCheckItem(
                check_name="OpenCV",
                status="ok" if opencv_available else "warn",
                summary="OpenCV is available for stronger registration and image processing." if opencv_available else "opencv-python is missing.",
                detail="cv2 import check",
                action=None if opencv_available else "Install opencv-python to enable stronger registration and pixel operations.",
            ),
            DesktopRuntimeCheckItem(
                check_name="Ultralytics",
                status="ok" if ultralytics_available else "warn",
                summary="Ultralytics is available for live detector training and evaluation." if ultralytics_available else "ultralytics is missing, so real detector runs stay in dry-run mode.",
                detail="ultralytics import check",
                action=None if ultralytics_available else "Install ultralytics or provide an external trainer command.",
            ),
            DesktopRuntimeCheckItem(
                check_name="PyTorch / CUDA",
                status="ok" if torch_available and cuda_available else ("partial" if torch_available else "warn"),
                summary="PyTorch with CUDA is ready for GPU-backed training." if torch_available and cuda_available else ("PyTorch is installed but CUDA is not available." if torch_available else "PyTorch is missing."),
                detail=gpu_summary,
                action=None if torch_available and cuda_available else ("Check CUDA runtime and GPU visibility." if torch_available else "Install PyTorch in the training environment."),
            ),
            DesktopRuntimeCheckItem(
                check_name="Packaged EXE",
                status="ok" if package_plan.exe_exists else "partial",
                summary="A Windows EXE build already exists." if package_plan.exe_exists else "The packaged EXE has not been generated yet.",
                detail=package_plan.exe_path,
                action=None if package_plan.exe_exists else "Run the EXE build script once after dependencies are ready.",
            ),
            DesktopRuntimeCheckItem(
                check_name="Program Status Report",
                status="ok" if self._report_exists(workspace_path, "reporting/program_status.json") else "partial",
                summary="Program status report exists and can drive guided UI state." if self._report_exists(workspace_path, "reporting/program_status.json") else "Program status report is missing.",
                detail=str(workspace_path / "evaluations" / "reporting" / "program_status.json"),
                action=None if self._report_exists(workspace_path, "reporting/program_status.json") else "Build Program Status once to populate the operator dashboard.",
            ),
            DesktopRuntimeCheckItem(
                check_name="Commercial Source Catalog",
                status="ok" if self._report_exists(workspace_path, "commercialization/source_catalog.json") else "partial",
                summary="Protected-source catalog exists for read-only intake." if self._report_exists(workspace_path, "commercialization/source_catalog.json") else "Protected-source catalog has not been built.",
                detail=str(workspace_path / "evaluations" / "commercialization" / "source_catalog.json"),
                action=None if self._report_exists(workspace_path, "commercialization/source_catalog.json") else "Build the Commercial Source Catalog to stage external-drive data safely.",
            ),
            DesktopRuntimeCheckItem(
                check_name="Review Queue",
                status="ok" if self._report_exists(workspace_path, "review_queue/report.json") else "partial",
                summary="Human-in-the-loop review queue exists." if self._report_exists(workspace_path, "review_queue/report.json") else "Review queue has not been built yet.",
                detail=str(workspace_path / "evaluations" / "review_queue" / "report.json"),
                action=None if self._report_exists(workspace_path, "review_queue/report.json") else "Build review queue from auto-label proposals before operator review starts.",
            ),
        ]

        readiness_score = min(
            100,
            sum(
                {
                    "Workspace Root": 10,
                    "UV Runtime": 6,
                    "PyInstaller": 10,
                    "OpenCV": 12,
                    "Ultralytics": 14,
                    "PyTorch / CUDA": 16,
                    "Packaged EXE": 10,
                    "Program Status Report": 8,
                    "Commercial Source Catalog": 7,
                    "Review Queue": 7,
                }.get(item.check_name, 0)
                * (1 if item.status == "ok" else 0.5 if item.status == "partial" else 0)
                for item in checks
            ),
        )

        blockers = [item.summary for item in checks if item.status == "warn"]
        next_actions = [item.action for item in checks if item.action][:6]

        response = DesktopRuntimeCheckResponse(
            workspace_root=str(workspace_path),
            readiness_score=int(round(readiness_score)),
            desktop_mode="desktop-exe" if package_plan.exe_exists else "browser-dev",
            python_executable=sys.executable,
            uv_available=uv_available,
            pyinstaller_ready=pyinstaller_ready,
            ultralytics_available=ultralytics_available,
            opencv_available=opencv_available,
            torch_available=torch_available,
            cuda_available=cuda_available,
            exe_exists=package_plan.exe_exists,
            package_build_ready=package_plan.build_ready,
            gpu_summary=gpu_summary,
            checks=checks,
            blockers=blockers,
            next_actions=next_actions,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> DesktopRuntimeCheckResponse:
        report_path = Path(workspace_root) / "evaluations" / "desktop_runtime" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing desktop runtime report: {report_path}")
        return DesktopRuntimeCheckResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _has_module(self, module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def _gpu_snapshot(self, torch_available: bool) -> tuple[bool, str]:
        if not torch_available:
            return False, "PyTorch not installed"
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
            if not cuda_available:
                return False, "PyTorch installed, CUDA unavailable"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count else "unknown"
            return True, f"CUDA ready ({device_count} GPU, primary={device_name})"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"PyTorch import failed: {exc.__class__.__name__}"

    def _report_exists(self, workspace_path: Path, relative_path: str) -> bool:
        return (workspace_path / "evaluations" / relative_path).exists()

    def _write_report(self, response: DesktopRuntimeCheckResponse) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
        )
        lines = [
            "# Desktop Runtime Diagnostics",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Readiness score: **{response.readiness_score}/100**",
            f"- Desktop mode: **{response.desktop_mode}**",
            f"- Python: `{response.python_executable}`",
            f"- GPU: `{response.gpu_summary}`",
            "",
            "## Checks",
        ]
        for item in response.checks:
            lines.append(f"- `{item.check_name}` status={item.status}")
            lines.append(f"  - summary: {item.summary}")
            if item.detail:
                lines.append(f"  - detail: {item.detail}")
            if item.action:
                lines.append(f"  - action: {item.action}")
        lines.extend(["", "## Blockers"])
        for item in response.blockers:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in response.next_actions:
            lines.append(f"- {item}")
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n")


desktop_runtime_service = DesktopRuntimeService()
