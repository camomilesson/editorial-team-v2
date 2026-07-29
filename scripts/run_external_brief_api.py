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
    await application.runtime_queue.start()
    loop = asyncio.get_running_loop()
    server = ExternalBriefHttpServer(
        (configuration.host, configuration.port),
        adapter=application.adapter,
        loop=loop,
    )
    trace_runtime_event("external_server_startup", correlation_id="external-server")
    try:
        await asyncio.to_thread(server.serve_forever)
    finally:
        application.adapter.stop_accepting()
        await asyncio.to_thread(server.shutdown)
        await asyncio.to_thread(server.server_close)
        await application.runtime_queue.close()
        trace_runtime_event("external_server_shutdown", correlation_id="external-server")


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
