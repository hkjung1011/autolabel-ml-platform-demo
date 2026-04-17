from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def runtime_is_frozen_bundle() -> bool:
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def _bundle_root() -> Path:
    if runtime_is_frozen_bundle() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _runtime_data_root() -> Path:
    if runtime_is_frozen_bundle():
        local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local_appdata / "DefectVisionResearch"
    return _bundle_root() / "demo_data"


@dataclass(frozen=True)
class Settings:
    app_name: str = "AutoLabel ML Platform Demo"
    version: str = "0.1.0"

    @property
    def root_dir(self) -> Path:
        return _project_root()

    @property
    def api_dir(self) -> Path:
        return _bundle_root()

    @property
    def static_dir(self) -> Path:
        if runtime_is_frozen_bundle():
            return self.api_dir / "app" / "static"
        return self.api_dir / "src" / "app" / "static"

    @property
    def demo_data_dir(self) -> Path:
        return _runtime_data_root()

    @property
    def upload_dir(self) -> Path:
        return self.demo_data_dir / "uploads"

    @property
    def variant_dir(self) -> Path:
        return self.demo_data_dir / "variants"

    def ensure_dirs(self) -> None:
        self.demo_data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.variant_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
