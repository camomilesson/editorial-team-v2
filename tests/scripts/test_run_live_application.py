"""Offline tests for the combined live executable."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_live_application.py"
_SPEC = importlib.util.spec_from_file_location("run_live_application", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Combined live script could not be loaded")
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_import_starts_nothing_and_main_runs_composed_polling(monkeypatch) -> None:
    events: list[str] = []

    class Telegram:
        def run_polling(self) -> None:
            events.append("polling")

    combined = SimpleNamespace(
        live=SimpleNamespace(
            model_name="safe-test-model",
            telegram=Telegram(),
        )
    )
    assert events == []
    monkeypatch.setattr(
        _SCRIPT,
        "build_combined_live_application_from_env",
        lambda: combined,
    )

    _SCRIPT.main()

    assert events == ["polling"]
