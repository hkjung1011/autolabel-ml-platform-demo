"""Post-build validation for the DefectVisionResearch PyInstaller bundle.

Layer 2 of the build-hygiene defense (L0 lock / L1 path split / L2 verify).
Guards against the stale-bytecode failure mode where the PyInstaller build
cache happily reuses bytecode from an older source tree even after edits.

Three verification layers run against a PyInstaller ``build/<name>/``
directory that has just been produced:

1. PYZ code symbols -- walk ``out00-PYZ.pyz`` with ``ZlibArchiveReader``,
   extract each required module's code object, and recursively collect every
   nested function/class name via ``co_consts``. Fails if a required symbol
   is missing from a rebuilt PYZ.

2. Source symbols for the entry script -- the entry script
   (``desktop_entry.py``) is stored as a ``PYSOURCE`` entry in the outer
   ``PKG-00.toc``, not in the PYZ. Locate its TOC entry, resolve the source
   path it references, and confirm the required function names exist via
   ``ast.parse``.

3. Bundled assets -- confirm each required data file (model weights, static
   HTML) is present in ``PKG-00.toc`` as a ``DATA`` entry with the expected
   destination path.

Exits 0 on success, 1 on any failure, with all error lines on stderr.
"""

from __future__ import annotations

import argparse
import ast
import sys
import types
from pathlib import Path
from typing import Iterable


REQUIRED_PYZ_SYMBOLS: dict[str, tuple[str, ...]] = {
    "app.core.atomic_io": (
        "atomic_write_text",
        "atomic_write_bytes",
        "atomic_write_json",
        "atomic_save_image",
    ),
    "app.services.training": (
        "_resolve_default_weights",
    ),
}

REQUIRED_SOURCE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "desktop_entry": (
        "_try_run_embedded_training",
        "_acquire_single_instance_mutex",
        "_write_instance_state",
    ),
}

REQUIRED_ASSETS: tuple[str, ...] = (
    "yolov8n.pt",
    "yolov8n-seg.pt",
    "yolo26n.pt",
    "app/static/index.html",
)

_KNOWN_TYPECODES: frozenset[str] = frozenset(
    {
        "PYMODULE",
        "PYSOURCE",
        "BINARY",
        "EXTENSION",
        "DATA",
        "RUNTIME_HOOK",
        "PYZ",
        "PKG",
        "EXECUTABLE",
        "SPLASH",
        "DEPENDENCY",
        "OPTION",
        "ZIPFILE",
    }
)


def _collect_code_symbols(code: types.CodeType) -> set[str]:
    """Walk ``co_consts`` recursively to gather every nested code object name."""
    symbols: set[str] = set()
    stack: list[types.CodeType] = [code]
    while stack:
        current = stack.pop()
        for const in current.co_consts:
            if isinstance(const, types.CodeType):
                symbols.add(const.co_name)
                stack.append(const)
    return symbols


def _load_pyz_code(pyz_path: Path, module: str) -> types.CodeType | None:
    """Extract a module's code object from the outer PyInstaller PYZ archive.

    PyInstaller versions differ in what ``ZlibArchiveReader.extract`` returns:
    newer builds hand back the code object directly, older builds return a
    ``(is_package, marshalled_bytes)`` tuple. Handle both forms.
    """
    try:
        from PyInstaller.archive.readers import ZlibArchiveReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyInstaller is not importable in this environment; "
            "verify_pyz.py must run inside the venv used to build the bundle"
        ) from exc
    reader = ZlibArchiveReader(str(pyz_path))
    if module not in reader.toc:
        return None
    entry = reader.extract(module)
    if isinstance(entry, types.CodeType):
        return entry
    if isinstance(entry, tuple) and len(entry) >= 2:
        payload = entry[1]
        if isinstance(payload, types.CodeType):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            import marshal

            code = marshal.loads(bytes(payload))
            if isinstance(code, types.CodeType):
                return code
    if isinstance(entry, (bytes, bytearray)):
        import marshal

        code = marshal.loads(bytes(entry))
        if isinstance(code, types.CodeType):
            return code
    return None


def _parse_pkg_toc(toc_path: Path) -> list[tuple]:
    """Parse PKG-00.toc (a Python literal) and return the bundle entries list."""
    raw = toc_path.read_text(encoding="utf-8")
    data = ast.literal_eval(raw)
    if not isinstance(data, tuple):
        raise ValueError(f"{toc_path.name}: top-level value is not a tuple")
    for item in data:
        if isinstance(item, list):
            return item
    raise ValueError(f"{toc_path.name}: no entries list found in top-level tuple")


def _normalize_dest(value: object) -> str:
    return str(value).replace("\\", "/")


def _entry_typecode(entry: tuple) -> str | None:
    if len(entry) >= 3 and isinstance(entry[2], str) and entry[2] in _KNOWN_TYPECODES:
        return entry[2]
    for value in entry[1:]:
        if isinstance(value, str) and value in _KNOWN_TYPECODES:
            return value
    return None


def _entry_source(entry: tuple) -> str | None:
    if len(entry) >= 2 and isinstance(entry[1], str):
        return entry[1]
    return None


def _find_entry(entries: Iterable[tuple], dest: str) -> tuple | None:
    target = dest.replace("\\", "/").lower()
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) < 1:
            continue
        entry_dest = _normalize_dest(entry[0]).lower()
        if entry_dest == target:
            return entry
    return None


def _parse_source_symbols(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _discover_pyz(build_dir: Path) -> Path | None:
    """Find the outer PYZ archive in a PyInstaller build directory.

    PyInstaller names this file differently across versions (``out00-PYZ.pyz``
    in older releases, ``PYZ-00.pyz`` in newer ones), so we just scan.
    """
    candidates = sorted(
        p
        for p in build_dir.glob("*.pyz")
        if p.is_file() and "base_library" not in p.name.lower()
    )
    return candidates[0] if candidates else None


def verify(build_dir: Path) -> list[str]:
    errors: list[str] = []

    pyz_path = _discover_pyz(build_dir)
    toc_path = build_dir / "PKG-00.toc"

    if pyz_path is None:
        errors.append(f"PYZ archive not found in {build_dir} (no matching *.pyz)")
    if not toc_path.is_file():
        errors.append(f"PKG TOC not found: {toc_path}")
    if errors:
        return errors
    assert pyz_path is not None  # for type checker after the guard above

    # Layer 1: PYZ code symbols
    for module, required in REQUIRED_PYZ_SYMBOLS.items():
        try:
            code = _load_pyz_code(pyz_path, module)
        except Exception as exc:
            errors.append(
                f"PYZ:{module}: extract failed ({type(exc).__name__}: {exc})"
            )
            continue
        if code is None:
            errors.append(f"PYZ:{module}: module not found in archive")
            continue
        symbols = _collect_code_symbols(code)
        missing = [name for name in required if name not in symbols]
        if missing:
            errors.append(f"PYZ:{module}: missing symbols {missing}")

    # Parse TOC once for layers 2 + 3
    try:
        entries = _parse_pkg_toc(toc_path)
    except Exception as exc:
        errors.append(f"PKG-00.toc: parse failed ({type(exc).__name__}: {exc})")
        return errors

    # Layer 2: Source symbols for entry script
    for module, required in REQUIRED_SOURCE_SYMBOLS.items():
        entry = _find_entry(entries, module)
        if entry is None:
            errors.append(f"PKG:{module}: entry not found in PKG TOC")
            continue
        typecode = _entry_typecode(entry)
        if typecode != "PYSOURCE":
            errors.append(
                f"PKG:{module}: expected PYSOURCE entry but got {typecode!r}"
            )
            continue
        source = _entry_source(entry)
        if not source:
            errors.append(f"PKG:{module}: PYSOURCE entry has no source path")
            continue
        source_path = Path(source)
        if not source_path.is_file():
            errors.append(
                f"PKG:{module}: source file referenced in TOC does not exist: {source_path}"
            )
            continue
        try:
            names = _parse_source_symbols(source_path)
        except Exception as exc:
            errors.append(
                f"PKG:{module}: ast.parse failed for {source_path} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        missing = [name for name in required if name not in names]
        if missing:
            errors.append(
                f"PKG:{module}: missing source symbols {missing} in {source_path}"
            )

    # Layer 3: Bundled assets
    for dest in REQUIRED_ASSETS:
        entry = _find_entry(entries, dest)
        if entry is None:
            errors.append(f"ASSET:{dest}: not found in PKG TOC")
            continue
        typecode = _entry_typecode(entry)
        if typecode != "DATA":
            errors.append(
                f"ASSET:{dest}: expected DATA entry but got {typecode!r}"
            )

    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "build_dir",
        type=Path,
        help="PyInstaller build/<name>/ directory containing out00-PYZ.pyz and PKG-00.toc",
    )
    args = parser.parse_args(argv)

    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        print(f"verify_pyz: build_dir does not exist: {build_dir}", file=sys.stderr)
        return 1

    errors = verify(build_dir)
    if errors:
        print(f"verify_pyz: {len(errors)} error(s) in {build_dir}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"verify_pyz: OK ({build_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
