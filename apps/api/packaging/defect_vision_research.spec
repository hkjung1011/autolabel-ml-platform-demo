# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

if not os.environ.get("DEFECT_VISION_BUILD_VIA_WRAPPER"):
    raise SystemExit("Use packaging/build.py instead of invoking PyInstaller directly.")

api_root = Path.cwd()
src_root = api_root / "src"
static_root = src_root / "app" / "static"
demo_root = api_root / "demo_data"
assets_root = api_root / "packaging" / "assets"


a = Analysis(
    [str(src_root / "app" / "desktop_entry.py")],
    pathex=[str(src_root)],
    binaries=[],
    datas=[
        (str(static_root), "app/static"),
        (str(demo_root), "demo_data"),
        (str(api_root / "yolo26n.pt"), "."),
        (str(api_root / "yolov8n.pt"), "."),
        (str(api_root / "yolov8n-seg.pt"), "."),
    ],
    hiddenimports=["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="DefectVisionResearch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(assets_root / "brand_mark.ico"),
    version=str(assets_root / "version_info.txt"),
)
