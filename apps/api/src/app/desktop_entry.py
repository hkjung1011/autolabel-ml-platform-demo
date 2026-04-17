from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
import time
import webbrowser
import ctypes
import sys
import urllib.request
from pathlib import Path

import uvicorn

from app.core.atomic_io import atomic_write_json
from app.core.config import runtime_is_frozen_bundle, settings
from app.main import app
from app.services.training import execute_embedded_training_request

_MUTEX_NAME = "Local\\DefectVisionResearchDesktopSingleton"
_MUTEX_ALREADY_EXISTS = 183
_DESKTOP_MUTEX_HANDLE = None


class _NullStream:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _ensure_standard_streams() -> None:
    if sys.stdout is None:
        sys.stdout = _NullStream()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullStream()  # type: ignore[assignment]


def _desktop_log_path() -> Path:
    log_dir = settings.demo_data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{_desktop_build_label()}_{os.getpid()}.log"


def _desktop_build_label() -> str:
    if runtime_is_frozen_bundle():
        return Path(sys.executable).stem
    return "desktop_dev"


def _instance_state_path() -> Path:
    desktop_dir = settings.demo_data_dir / "desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    return desktop_dir / "active_instance.json"


def _acquire_single_instance_mutex() -> tuple[object | None, bool]:
    kernel32 = getattr(ctypes, "windll", None)
    if kernel32 is None:
        return None, False
    handle = kernel32.kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if not handle:
        return None, False
    already_exists = kernel32.kernel32.GetLastError() == _MUTEX_ALREADY_EXISTS
    return handle, already_exists


def _release_single_instance_mutex(handle: object | None) -> None:
    if handle is None:
        return
    kernel32 = getattr(ctypes, "windll", None)
    if kernel32 is None:
        return
    try:
        kernel32.kernel32.CloseHandle(handle)
    except Exception:
        return


def _load_instance_state() -> dict[str, object] | None:
    state_path = _instance_state_path()
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8").strip() or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_instance_state(*, port: int, log_path: Path, ready: bool) -> None:
    atomic_write_json(
        _instance_state_path(),
        {
            "pid": os.getpid(),
            "port": port,
            "ready": ready,
            "build_label": _desktop_build_label(),
            "version": settings.version,
            "frozen": runtime_is_frozen_bundle(),
            "executable": sys.executable,
            "log_path": str(log_path),
            "updated_at": time.time(),
        },
    )


def _clear_instance_state() -> None:
    state = _load_instance_state()
    if not state or state.get("pid") != os.getpid():
        return
    try:
        _instance_state_path().unlink(missing_ok=True)
    except Exception:
        return


def _wait_for_existing_instance(timeout_seconds: float = 10.0) -> dict[str, object] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = _load_instance_state()
        if state:
            port = int(state.get("port", 0) or 0)
            if port > 0 and _is_http_ready(port):
                return state
        time.sleep(0.2)
    return None


def _configure_desktop_logging() -> Path:
    _ensure_standard_streams()
    log_path = _desktop_log_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def _try_run_training_script() -> bool:
    """Execute ``sys.argv[1]`` as a Python script if it looks like one.

    The subprocess training path in ``training.py`` spawns the frozen exe with a
    runner script as ``argv[1]``. In frozen mode ``sys.executable`` points at the
    exe itself, so without this early dispatch ``main()`` would ignore the
    argument and launch another uvicorn server instead of running the training
    script. ``stdout``/``stderr`` are inherited from the parent ``Popen`` call,
    which redirects them to ``live.log`` in the run output directory.
    """
    if len(sys.argv) < 2:
        return False
    candidate = sys.argv[1]
    if not candidate.lower().endswith(".py"):
        return False
    script_path = Path(candidate)
    if not script_path.is_file():
        return False

    _ensure_standard_streams()
    print(f"[desktop_entry] runner mode: executing {script_path}", flush=True)
    try:
        import runpy

        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        print(f"[desktop_entry] runner SystemExit code={exc.code}", flush=True)
    except BaseException as exc:
        import traceback

        print(
            f"[desktop_entry] runner failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"[desktop_entry] runner finished: {script_path}", flush=True)
    return True


def _try_run_embedded_training() -> bool:
    if len(sys.argv) < 3 or sys.argv[1] != "--embedded-train":
        return False
    _ensure_standard_streams()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("desktop_train_entry")
    request_path = sys.argv[2]
    logger.info("Starting embedded training run from %s", request_path)
    exit_code = execute_embedded_training_request(request_path)
    logger.info("Embedded training finished with exit code %s", exit_code)
    raise SystemExit(exit_code)


def _schedule_server_shutdown(server: uvicorn.Server) -> None:
    logger = logging.getLogger("desktop_entry")

    def _shutdown() -> None:
        time.sleep(0.25)
        logger.info("Desktop shutdown requested; stopping server.")
        _clear_instance_state()
        server.should_exit = True
        server.force_exit = True

    threading.Thread(target=_shutdown, daemon=True).start()


def main() -> int:
    global _DESKTOP_MUTEX_HANDLE
    if _try_run_embedded_training():
        return 0
    if _try_run_training_script():
        return 0
    mutex_handle, already_exists = _acquire_single_instance_mutex()
    _DESKTOP_MUTEX_HANDLE = mutex_handle
    if already_exists:
        existing = _wait_for_existing_instance()
        if existing is not None:
            port = int(existing.get("port", 0) or 0)
            if port > 0:
                webbrowser.open(f"http://127.0.0.1:{port}/?mode=desktop&attached=1")
        _release_single_instance_mutex(_DESKTOP_MUTEX_HANDLE)
        _DESKTOP_MUTEX_HANDLE = None
        return 0

    log_path = _configure_desktop_logging()
    logger = logging.getLogger("desktop_entry")
    try:
        ctypes.windll.kernel32.SetConsoleTitleW("Defect Vision Operator Launcher")
    except Exception:
        pass
    port = _find_free_port()
    os.environ["DEFECT_VISION_PORT"] = str(port)
    os.environ["DEFECT_VISION_BUILD_LABEL"] = _desktop_build_label()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    app.state.desktop_shutdown_callback = lambda: _schedule_server_shutdown(server)
    thread = threading.Thread(target=server.run, daemon=False)
    logger.info("Starting desktop launcher on http://127.0.0.1:%s (log=%s)", port, log_path)
    _write_instance_state(port=port, log_path=log_path, ready=False)
    atexit.register(_clear_instance_state)
    thread.start()

    ready = False
    for _ in range(600):
        if getattr(server, "started", False) and _is_http_ready(port):
            ready = True
            break
        time.sleep(0.1)

    if ready:
        _write_instance_state(port=port, log_path=log_path, ready=True)
        webbrowser.open(f"http://127.0.0.1:{port}/?mode=desktop")
        logger.info("Browser open requested for operator UI.")
    else:
        logger.error("Desktop launcher timed out waiting for HTTP readiness on port %s.", port)

    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5.0)
        logger.info("Desktop launcher interrupted and stopped.")
    finally:
        _clear_instance_state()
        _release_single_instance_mutex(_DESKTOP_MUTEX_HANDLE)
        _DESKTOP_MUTEX_HANDLE = None
    return 0


def _is_http_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
            return response.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
