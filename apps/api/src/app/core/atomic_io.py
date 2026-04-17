from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.write_text(content, encoding=encoding)
    os.replace(tmp_path, target)


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.write_bytes(content)
    os.replace(tmp_path, target)


def atomic_write_json(path: str | Path, payload: object, *, ensure_ascii: bool = False, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
        encoding="utf-8",
    )


def atomic_save_image(path: str | Path, image: Image.Image, *, format: str | None = None) -> None:
    target = Path(path)
    buffer = BytesIO()
    save_format = format or (target.suffix.lstrip(".").upper() if target.suffix else None) or "PNG"
    image.save(buffer, format=save_format)
    atomic_write_bytes(target, buffer.getvalue())
