#!/usr/bin/env python3
"""Run the authenticated external brief HTTP server."""

from __future__ import annotations

import asyncio
import logging

from editorial_team.app import (
    ExternalApiConfigurationError,
    LiveConfigurationError,
    build_external_api_application,
    load_external_api_configuration,
)
from editorial_team.interfaces.external_http import ExternalBriefHttpServer
from editorial_team.tracing import trace_runtime_event


async def serve() -> None:
    """Start the shared runtime and serve until interrupted."""

    configuration = load_external_api_configuration()
    application = build_external_api_application(configuration.token)
    loop = asyncio.get_running_loop()
    try:
        server = ExternalBriefHttpServer(
            (configuration.host, configuration.port),
            adapter=application.adapter,
            loop=loop,
        )
    except Exception:
        raise LiveConfigurationError("External HTTP server configuration is invalid") from None
    try:
        await application.runtime_queue.start()
    except Exception:
        await asyncio.to_thread(server.server_close)
        await application.runtime_queue.close()
        raise LiveConfigurationError("External API runtime could not start") from None
    try:
        trace_runtime_event("external_server_started", correlation_id="external-server")
        await asyncio.to_thread(server.serve_forever)
    finally:
        application.adapter.stop_accepting()
        await asyncio.to_thread(server.shutdown)
        await asyncio.to_thread(server.server_close)
        await application.runtime_queue.close()
        trace_runtime_event("external_server_stopped", correlation_id="external-server")


def main() -> None:
    """Load configuration and run the server."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )
    try:
        asyncio.run(serve())
    except (ExternalApiConfigurationError, LiveConfigurationError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
