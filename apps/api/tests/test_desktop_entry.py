import logging
import sys
from pathlib import Path

import pytest

from app import desktop_entry


def test_configure_desktop_logging_handles_missing_streams(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "desktop_runtime.log"

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(desktop_entry, "_desktop_log_path", lambda: log_path)

    configured_path = desktop_entry._configure_desktop_logging()

    assert configured_path == log_path
    assert sys.stdout is not None
    assert sys.stderr is not None

    logger = logging.getLogger("desktop_entry_test")
    logger.info("desktop logging smoke")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.exists()
    assert "desktop logging smoke" in log_path.read_text(encoding="utf-8")


def test_embedded_training_flag_dispatches_runner(tmp_path: Path, monkeypatch) -> None:
    request_path = tmp_path / "embedded_request.json"
    request_path.write_text("{}", encoding="utf-8")
    called: dict[str, str] = {}

    def _fake_runner(path: str) -> int:
        called["path"] = path
        return 0

    monkeypatch.setattr(desktop_entry, "execute_embedded_training_request", _fake_runner)
    monkeypatch.setattr(sys, "argv", ["DefectVisionResearch.exe", "--embedded-train", str(request_path)])

    with pytest.raises(SystemExit) as exc:
        desktop_entry._try_run_embedded_training()

    assert exc.value.code == 0
    assert called["path"] == str(request_path)


def test_main_attaches_to_existing_instance(monkeypatch) -> None:
    opened: dict[str, str] = {}

    monkeypatch.setattr(desktop_entry, "_try_run_embedded_training", lambda: False)
    monkeypatch.setattr(desktop_entry, "_try_run_training_script", lambda: False)
    monkeypatch.setattr(desktop_entry, "_acquire_single_instance_mutex", lambda: ("mutex", True))
    monkeypatch.setattr(desktop_entry, "_wait_for_existing_instance", lambda: {"port": 58053})
    monkeypatch.setattr(desktop_entry, "_release_single_instance_mutex", lambda handle: opened.setdefault("released", str(handle)))
    monkeypatch.setattr(desktop_entry.webbrowser, "open", lambda url: opened.setdefault("url", url))

    assert desktop_entry.main() == 0
    assert opened["url"] == "http://127.0.0.1:58053/?mode=desktop&attached=1"
    assert opened["released"] == "mutex"
