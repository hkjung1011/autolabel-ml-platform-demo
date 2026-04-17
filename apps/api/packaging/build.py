"""Build wrapper for the DefectVisionResearch PyInstaller bundle.

Layers 0 and 1 of the build-hygiene defense:

- **L0 build lock** -- a file-based exclusive lock at ``apps/api/.build_lock``
  prevents two concurrent builds (e.g. Claude + Codex) from stomping on each
  other. A stale lock left behind by a crashed build is auto-cleared after a
  PID-liveness check.

- **L1 path split** -- each build agent gets its own ``build-staging/<agent>``
  (workpath) and ``dist-staging/<agent>`` (distpath) so PyInstaller's build
  cache cannot silently hand one agent stale bytecode written by the other.
  The agent name comes from ``--agent`` (or ``$BUILD_AGENT``, default
  ``claude``).

After PyInstaller succeeds this script unconditionally runs ``verify_pyz.py``
(Layer 2) against the freshly written workpath, and only promotes the
staging dist into the canonical ``dist/DefectVisionResearch.exe`` if verify
returns 0. A crashed promote rolls back to the previous dist via a
``DefectVisionResearch.prev.exe`` rename.

Direct invocation of the spec (``pyinstaller packaging/defect_vision_research.spec``)
is blocked by a top-of-spec guard that checks for the
``DEFECT_VISION_BUILD_VIA_WRAPPER`` env var this script sets.

Usage::

    python packaging/build.py [--agent NAME] [--clean]

Exit codes:
    0   Build + verify + promotion succeeded.
    1   Any step failed (lock busy, PyInstaller error, verify error, promote error).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = API_ROOT / "packaging" / "defect_vision_research.spec"
VERIFY_SCRIPT = API_ROOT / "packaging" / "verify_pyz.py"
LOCK_PATH = API_ROOT / ".build_lock"
SPEC_NAME = "defect_vision_research"
EXE_NAME = "DefectVisionResearch"
VENV_PYTHON = API_ROOT / ".venv" / "Scripts" / "python.exe"


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        pid: int | None
        try:
            content = lock_path.read_text(encoding="utf-8").strip()
            pid = int(content.split()[0])
        except (OSError, ValueError, IndexError):
            pid = None
        if pid and _pid_alive(pid):
            raise RuntimeError(
                f"Build lock held by PID {pid} (lock file: {lock_path}). "
                "Wait for the other build to finish or remove the lock file manually."
            )
        lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Build lock race: another process acquired {lock_path} just now"
        ) from exc
    try:
        os.write(fd, f"{os.getpid()} {int(time.time())}".encode("utf-8"))
    finally:
        os.close(fd)


def _release_lock(lock_path: Path) -> None:
    try:
        content = lock_path.read_text(encoding="utf-8").strip()
        pid = int(content.split()[0])
    except (OSError, ValueError, IndexError):
        return
    if pid == os.getpid():
        lock_path.unlink(missing_ok=True)


def _resolve_agent(cli_agent: str | None) -> str:
    if cli_agent:
        return cli_agent
    env_agent = os.environ.get("BUILD_AGENT")
    if env_agent:
        return env_agent
    return "claude"


def _rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _run_pyinstaller(
    *,
    python_executable: Path,
    distpath: Path,
    workpath: Path,
    spec_path: Path,
    env: dict[str, str],
) -> int:
    cmd = [
        str(python_executable),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(distpath),
        "--workpath",
        str(workpath),
        str(spec_path),
    ]
    print(f"[build] invoking PyInstaller: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(API_ROOT), env=env)
    return proc.returncode


def _run_verify(work_build_dir: Path) -> int:
    python_executable = _resolve_build_python()
    cmd = [str(python_executable), str(VERIFY_SCRIPT), str(work_build_dir)]
    print(f"[build] running verify: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(API_ROOT))
    return proc.returncode


def _promote(staging_dist_root: Path, canonical_dist_root: Path) -> None:
    staging_exe_path = staging_dist_root / f"{EXE_NAME}.exe"
    if not staging_exe_path.is_file():
        raise RuntimeError(f"Staging executable missing: {staging_exe_path}")
    canonical_dist_root.mkdir(parents=True, exist_ok=True)
    canonical_exe_path = canonical_dist_root / f"{EXE_NAME}.exe"
    backup_path = canonical_dist_root / f"{EXE_NAME}.prev.exe"

    backup_path.unlink(missing_ok=True)
    restored = False
    if canonical_exe_path.exists():
        canonical_exe_path.rename(backup_path)
    try:
        shutil.move(str(staging_exe_path), str(canonical_exe_path))
    except Exception:
        if backup_path.exists():
            canonical_exe_path.unlink(missing_ok=True)
            backup_path.rename(canonical_exe_path)
            restored = True
        raise
    finally:
        if not restored and backup_path.exists() and canonical_exe_path.exists():
            backup_path.unlink(missing_ok=True)


def _resolve_build_python() -> Path:
    if VENV_PYTHON.is_file():
        return VENV_PYTHON
    return Path(sys.executable)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default=None,
        help="Agent namespace for workpath/distpath split (default: $BUILD_AGENT or 'claude')",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the staging workpath/distpath before building",
    )
    args = parser.parse_args(argv)

    if not SPEC_PATH.is_file():
        print(f"[build] spec not found: {SPEC_PATH}", file=sys.stderr)
        return 1
    if not VERIFY_SCRIPT.is_file():
        print(f"[build] verify_pyz.py not found: {VERIFY_SCRIPT}", file=sys.stderr)
        return 1

    agent = _resolve_agent(args.agent)
    workpath = API_ROOT / "build-staging" / agent
    distpath = API_ROOT / "dist-staging" / agent
    canonical_dist_root = API_ROOT / "dist"

    try:
        _acquire_lock(LOCK_PATH)
    except RuntimeError as exc:
        print(f"[build] {exc}", file=sys.stderr)
        return 1

    try:
        if args.clean:
            print(
                f"[build] --clean: removing {workpath} and {distpath}", flush=True
            )
            _rmtree(workpath)
            _rmtree(distpath)

        workpath.mkdir(parents=True, exist_ok=True)
        distpath.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["DEFECT_VISION_BUILD_VIA_WRAPPER"] = "1"
        env["BUILD_AGENT"] = agent
        python_executable = _resolve_build_python()

        print(
            f"[build] agent={agent} workpath={workpath} distpath={distpath} python={python_executable}",
            flush=True,
        )

        rc = _run_pyinstaller(
            python_executable=python_executable,
            distpath=distpath,
            workpath=workpath,
            spec_path=SPEC_PATH,
            env=env,
        )
        if rc != 0:
            print(f"[build] PyInstaller failed with exit code {rc}", file=sys.stderr)
            return 1

        work_build_dir = workpath / SPEC_NAME
        rc = _run_verify(work_build_dir)
        if rc != 0:
            print(
                f"[build] verify_pyz failed (exit {rc}); "
                f"staging dist NOT promoted to {canonical_dist_root}",
                file=sys.stderr,
            )
            return 1

        try:
            _promote(distpath, canonical_dist_root)
        except Exception as exc:
            print(f"[build] promote failed: {exc}", file=sys.stderr)
            return 1

        print(
            f"[build] OK: promoted staging -> {canonical_dist_root / f'{EXE_NAME}.exe'}",
            flush=True,
        )
        return 0
    finally:
        _release_lock(LOCK_PATH)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
