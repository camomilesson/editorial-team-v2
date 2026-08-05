#!/usr/bin/env python3
"""Run Telegram and heartbeat as one shared live application."""

from __future__ import annotations

import logging

from editorial_team.app import (
    RECENT_MESSAGE_LIMIT,
    LiveConfigurationError,
    build_live_application_from_env,
)


def main() -> None:
    """Compose and run the live application until interrupted."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        live = build_live_application_from_env()
    except LiveConfigurationError as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(2) from None

    logging.getLogger(__name__).info(
        "live_application_ready model=%s conversation_persistence=sqlite "
        "recent_message_limit=%d processing=one-shared-runtime-queue",
        live.model_name,
        RECENT_MESSAGE_LIMIT,
    )
    live.telegram.run_polling()


if __name__ == "__main__":
    main()
