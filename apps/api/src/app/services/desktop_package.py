from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from app.core.atomic_io import atomic_write_text
from app.core.config import runtime_is_frozen_bundle, settings
from app.domain.research_models import DesktopPackagePlanResponse


class DesktopPackageService:
    def build_plan(self, workspace_root: str) -> DesktopPackagePlanResponse:
        workspace_path = Path(workspace_root)
        bundled_mode = runtime_is_frozen_bundle()
        packaging_root = settings.api_dir / "packaging"
        scripts_root = settings.api_dir / "scripts"
        entry_script = settings.api_dir / "src" / "app" / "desktop_entry.py"
        spec_path = packaging_root / "defect_vision_research.spec"
        build_script = scripts_root / "build_exe.ps1"
        dist_dir = Path(sys.executable).parent if bundled_mode else settings.api_dir / "dist"
        exe_path = Path(sys.executable) if bundled_mode else dist_dir / "DefectVisionResearch.exe"
        exe_exists = exe_path.exists()

        pyinstaller_ready = True if bundled_mode else bool(importlib.util.find_spec("PyInstaller"))
        blockers: list[str] = []
        if not pyinstaller_ready:
            blockers.append("PyInstaller is not installed in the current environment.")
        if not bundled_mode:
            for required_path, label in [
                (entry_script, "desktop entry script"),
                (spec_path, "PyInstaller spec"),
                (build_script, "PowerShell build script"),
            ]:
                if not required_path.exists():
                    blockers.append(f"Missing {label}: {required_path}")

        command_preview = (
            "Packaged EXE is already running; rebuild from the development environment if needed."
            if bundled_mode
            else f'powershell -ExecutionPolicy Bypass -File "{build_script}"'
        )
        design_notes = [
            "The packaged exe should launch the local FastAPI server and open the UI in the default browser.",
            "Desktop mode should open with a branded operator console state rather than a generic browser-like landing page.",
            "Keep the current static UI as the packaged shell until the core workflow is stable enough for a React migration.",
            "Preserve operator session values across launches so pilot users do not re-enter workspace and scan roots every time.",
            "Package assets, static files, and demo directories through the PyInstaller spec so the UI looks identical inside and outside development mode.",
        ]
        next_actions: list[str] = []
        if not pyinstaller_ready:
            next_actions.append("Install PyInstaller in the packaging environment.")
        if not exe_exists:
            next_actions.append("Run the build script once to generate the first Windows exe.")
        else:
            next_actions.append("Smoke-test the packaged exe on a clean Windows machine.")
        next_actions.append("Add an application icon and metadata before distributing the first exe build.")
        next_actions.append("Validate desktop-mode launch and session restore behavior in the packaged exe.")
        report_root = workspace_path / "evaluations" / "packaging"
        report_root.mkdir(parents=True, exist_ok=True)
        response = DesktopPackagePlanResponse(
            workspace_root=str(workspace_path),
            app_root=str(Path(sys.executable).parent if bundled_mode else settings.api_dir),
            entry_script_path=(f"{sys.executable} [embedded desktop entry]" if bundled_mode else str(entry_script)),
            build_script_path=("embedded build metadata" if bundled_mode else str(build_script)),
            spec_path=("embedded packaging metadata" if bundled_mode else str(spec_path)),
            dist_dir=str(dist_dir),
            exe_path=str(exe_path),
            exe_exists=exe_exists,
            pyinstaller_ready=pyinstaller_ready,
            build_ready=pyinstaller_ready and not blockers,
            command_preview=command_preview,
            blockers=blockers,
            design_notes=design_notes,
            next_actions=next_actions,
            report_json_path=str(report_root / "plan.json"),
            report_markdown_path=str(report_root / "plan.md"),
        )
        self._write_report(response)
        return response

    def load_plan(self, workspace_root: str) -> DesktopPackagePlanResponse:
        report_path = Path(workspace_root) / "evaluations" / "packaging" / "plan.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing desktop packaging plan: {report_path}")
        payload = report_path.read_text(encoding="utf-8").strip()
        if not payload:
            raise FileNotFoundError(f"Empty desktop packaging plan: {report_path}")
        try:
            return DesktopPackagePlanResponse.model_validate_json(payload)
        except ValidationError as exc:
            raise FileNotFoundError(f"Corrupt desktop packaging plan: {report_path}") from exc

    def _write_report(self, response: DesktopPackagePlanResponse) -> None:
        atomic_write_text(
            response.report_json_path,
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
        )
        lines = [
            "# Desktop Packaging Plan",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- App root: `{response.app_root}`",
            f"- Entry script: `{response.entry_script_path}`",
            f"- Spec path: `{response.spec_path}`",
            f"- Build script: `{response.build_script_path}`",
            f"- Dist dir: `{response.dist_dir}`",
            f"- EXE path: `{response.exe_path}`",
            f"- EXE exists: **{response.exe_exists}**",
            f"- PyInstaller ready: **{response.pyinstaller_ready}**",
            f"- Build ready: **{response.build_ready}**",
            f"- Command: `{response.command_preview}`",
            "",
            "## Blockers",
        ]
        for blocker in response.blockers:
            lines.append(f"- {blocker}")
        lines.extend(["", "## Design Notes"])
        for item in response.design_notes:
            lines.append(f"- {item}")
        lines.extend(["", "## Next Actions"])
        for item in response.next_actions:
            lines.append(f"- {item}")
        atomic_write_text(response.report_markdown_path, "\n".join(lines) + "\n")


desktop_package_service = DesktopPackageService()
