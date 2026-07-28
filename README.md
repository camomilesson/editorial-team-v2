# Editorial Team v2

Provider-neutral infrastructure for a conversational editorial assistant.

This foundation contains model-client boundaries, a Gemini adapter, validation helpers,
sanitized errors, and a generic tool registry. Product behavior is intentionally out of scope.

## Development

Requires Python 3.11 or newer.

```shell
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Use `.env.example` as a configuration reference and provide `GEMINI_API_KEY` through
the process environment to use the Gemini adapter.

## Local Telegram bot

The live adapter supports private Telegram chats using long polling. Group chats and
non-text updates are deliberately ignored in this first slice.

Provide `GEMINI_API_KEY`, `AGENT_MODEL`, and `TELEGRAM_BOT_TOKEN` through the process
environment supplied by your shell or editor. The application does not load `.env` files.

```shell
python scripts/run_telegram_bot.py
```

Conversation state is in memory: restarting the process loses conversations and active
tasks. Until the later event-queue milestone, Telegram turns are deliberately serialized
with one in-flight turn at a time. This causes head-of-line blocking when a model call is
slow; it is not the final queue design.

### Manual smoke test

1. Start the bot locally with the command above.
2. Open the disposable bot in Telegram.
3. Send `Hello!` and expect one normal conversational response.
4. Send `Write a short LinkedIn post announcing that Editorial Team is now available as a
   Telegram bot.` Expect Writer output, a Critic evaluation, either a pass confirmation or
   Editor revision, and a request for evaluation.
5. Send `Make it shorter and less formal.` Expect another normal
   Writer–Critic–optional Editor cycle using the current working draft, followed by a new
   evaluation request.
6. Send `Looks good, thanks.` Expect approval and a conversational acknowledgement.
7. Stop and restart the process. The prior state should be gone because storage is
   intentionally in memory.

This smoke test is manual and is not performed by the automated test suite.
