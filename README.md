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
tasks.

## Runtime queue

One application-owned, bounded FIFO queue is the execution boundary for runtime work.
Telegram turns currently enter this queue; later webhook and heartbeat producers will use
the same boundary. One asynchronous worker executes accepted jobs in deterministic order,
isolates individual failures, and drains accepted work during graceful shutdown. Waiting
capacity defaults to 100. The queue has no persistence or retries and does not survive a
process restart. The deliberate trade-off is head-of-line blocking when one LLM workflow
is slow in exchange for simple, serialized access to in-memory conversation state.
Queue tracing measures waiting depth immediately after enqueue and after dequeue for
start/completion events; the currently executing job is not included in that depth.

The Coordinator routes messages to `CHAT`, `START_WRITING_TASK`, or `REVISE_TASK`.
There is no explicit approval phase: the latest completed writing task remains available
for later revision, including after intervening chat, until a new writing request replaces
it. Writer, Critic, and Editor are displayed as a staged Telegram handoff after the complete
Writer–Critic–optional Editor workflow has finished atomically. This presentation delay is
not real-time workflow streaming.

### Manual smoke test

1. Start the bot locally with the command above.
2. Open the disposable bot in Telegram.
3. Send `Hello!` and expect one normal conversational response.
4. Send `Write a short LinkedIn post announcing that Editorial Team is now available as a
   Telegram bot.` Expect Writer output, a Critic evaluation, and an Editor handoff message.
5. Send `Make it shorter and less formal.` Expect another normal
   Writer–Critic–optional Editor cycle using the current working draft.
6. Send `Looks good, thanks.` Expect a conversational acknowledgement while the latest task
   remains available for revision.
7. Stop and restart the process. The prior state should be gone because storage is
   intentionally in memory.

This smoke test is manual and is not performed by the automated test suite.

## Operational decision storage

SQLite has one narrow role: durable storage of heartbeat/Admin decisions.
Both `SILENCE` and `NOTIFY` outcomes are stored. `SILENCE` produces no Telegram
output; `NOTIFY` alerts one configured maintainer destination and records whether
that notification was sent.

The default local path is `runtime_data/editorial_team.db`, but callers inject the
database path explicitly. Live asynchronous orchestration calls the synchronous
repository through explicit `asyncio.to_thread` boundaries.

This database stores no user messages, identities, prompts, conversations, writing
tasks, drafts, Critic reports, or raw model output. It is not queue persistence:
the runtime queue and normal Telegram conversation state remain in memory and reset
when the process restarts.

### Privileged AdminAgent

AdminAgent is an operational subagent that receives only an `OperationalSnapshot`
and an immutable threshold policy. It chooses `SILENCE` or `NOTIFY`; application
code independently applies the same deterministic policy and rejects conflicting
model output before storing a result in SQLite.

The default policy checks conditions in strict priority order: stopped worker,
three or more failed jobs, queue occupancy at or above 0.8, then system healthy.
AdminAgent cannot access conversations, user identities, editorial prompts, or
drafts. It cannot send Telegram messages, write SQLite directly, or modify runtime
state.

### Optional live heartbeat

The in-process heartbeat is disabled by default. Enable it with
`EDITORIAL_HEARTBEAT_ENABLED=true` and configure
`EDITORIAL_ADMIN_TELEGRAM_CHAT_ID`. The interval defaults to 900 seconds
(15 minutes) and may be changed with `EDITORIAL_HEARTBEAT_INTERVAL_SECONDS`;
live configuration rejects intervals below 10 seconds. The injected SQLite path
defaults to `runtime_data/editorial_team.db` and can be changed with
`EDITORIAL_HEARTBEAT_DB_PATH`.

Each interval, the collector records waiting depth and calculates completed and
failed `TELEGRAM` and `EXTERNAL` job deltas since the previous observation.
`HEARTBEAT` jobs are excluded from their own observation window. The first
observation counts relevant jobs since process startup. The heartbeat then joins
the same bounded FIFO queue as user work, so it cannot interleave with a staged
editorial workflow.

The fixed Admin priority remains stopped worker, three or more recent failures,
queue occupancy of at least 0.8, then healthy. Both outcomes are persisted.
`SILENCE` sends nothing and remains `notification_sent=false`. `NOTIFY` sends
deterministic, non-LLM prose to the configured maintainer chat and is marked sent
only after successful delivery. Delivery or persistence failures are not retried.
SQLite continues to store no chat ID, identity, message, prompt, or draft data.

Because the scheduled runner depends on the same in-process queue worker,
complete worker or process loss cannot be independently detected by this path.
`WORKER_STOPPED` remains supported by policy and injected/forced evaluations; an
external watchdog would be required for independent loss detection.

Inspect recent operational decisions locally:

```shell
sqlite3 runtime_data/editorial_team.db \
  "SELECT observed_at, decision, reason_code, notification_sent
   FROM heartbeat_results
   ORDER BY observed_at DESC
   LIMIT 10;"
```

## Structured-output reliability

The Coordinator and Critic request provider-native, JSON Schema-constrained output.
Their responses still pass through the application's strict JSON parser, domain
validation, and, for Critic issues, grounded-excerpt validation. The Talker, Writer,
and Editor continue to return plain text. Schema constraints improve formatting
reliability but do not make semantic agent output infallible.
