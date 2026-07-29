"""Offline lifecycle tests for the external brief executable."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_external_brief_api.py"
_SPEC = importlib.util.spec_from_file_location("run_external_brief_api", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("External brief script could not be loaded")
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_import_starts_no_runtime_and_serve_closes_all_lifecycle_components(
    monkeypatch,
) -> None:
    events: list[str] = []

    class Queue:
        async def start(self) -> None:
            events.append("queue-start")

        async def close(self) -> None:
            events.append("queue-close")

    class Adapter:
        def stop_accepting(self) -> None:
            events.append("stop-accepting")

    class Server:
        def __init__(self, address, *, adapter, loop) -> None:
            assert address == ("127.0.0.1", 8080)
            assert adapter is application.adapter
            assert loop is not None
            events.append("server-created")

        def serve_forever(self) -> None:
            events.append("serve")

        def shutdown(self) -> None:
            events.append("shutdown")

        def server_close(self) -> None:
            events.append("server-close")

    application = SimpleNamespace(runtime_queue=Queue(), adapter=Adapter())
    configuration = SimpleNamespace(
        token="placeholder-token",
        host="127.0.0.1",
        port=8080,
    )
    assert events == []
    monkeypatch.setattr(
        _SCRIPT, "load_external_api_configuration", lambda: configuration
    )
    monkeypatch.setattr(
        _SCRIPT, "build_external_api_application", lambda token: application
    )
    monkeypatch.setattr(_SCRIPT, "ExternalBriefHttpServer", Server)

    asyncio.run(_SCRIPT.serve())

    assert events == [
        "queue-start",
        "server-created",
        "serve",
        "stop-accepting",
        "shutdown",
        "server-close",
        "queue-close",
    ]
