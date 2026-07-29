#!/usr/bin/env python3
"""Run the local Telegram bot using long polling."""

from __future__ import annotations

import logging

from editorial_team.app import (
    RECENT_MESSAGE_LIMIT,
    LiveConfigurationError,
    build_live_application_from_env,
)


def main() -> None:
    """Compose and run the live bot until interrupted."""

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
        "telegram_bot_started model=%s conversation_persistence=in-memory "
        "recent_message_limit=%d "
        "processing=one-in-flight-turn",
        live.model_name,
        RECENT_MESSAGE_LIMIT,
    )
    live.telegram.run_polling()


if __name__ == "__main__":
    main()
