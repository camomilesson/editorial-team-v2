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

## Combined live application

The recommended class-demo runtime starts Telegram long polling, the external brief HTTP
API, and the optional heartbeat in one process over exactly one shared `RuntimeQueue`.
Heartbeat therefore observes both `TELEGRAM` and `EXTERNAL` activity.

Provide `GEMINI_API_KEY`, `AGENT_MODEL`, `TELEGRAM_BOT_TOKEN`, and a non-empty
`EDITORIAL_EXTERNAL_API_TOKEN` through the process environment supplied by your shell,
editor, or the real ignored `.env`. `.env.example` intentionally leaves credentials blank;
missing or blank required configuration stops startup safely. The application does not
load `.env` files itself.

```shell
python scripts/run_live_application.py
```

Conversation state is in memory: restarting the process loses conversations and active
tasks.

## Standalone external brief API

The external brief server exposes one synchronous authenticated endpoint that sends a
standalone writing brief through the Writer–Critic–Editor workflow. This focused
development/testing command runs in a separate process with its own process-local queue and
metrics. It does not start Telegram polling or heartbeat, so Telegram heartbeat cannot
observe its `EXTERNAL` activity.

Provide `GEMINI_API_KEY`, `AGENT_MODEL`, and a non-empty
`EDITORIAL_EXTERNAL_API_TOKEN` through the process environment. The server defaults to
`127.0.0.1:8080`; `EDITORIAL_EXTERNAL_API_HOST` and
`EDITORIAL_EXTERNAL_API_PORT` may override that address.

```shell
python scripts/run_external_brief_api.py
```

Send JSON with a bearer token and a non-empty idempotency key:

```shell
curl -X POST http://127.0.0.1:8080/brief \
  -H "Authorization: Bearer placeholder-local-token" \
  -H "Idempotency-Key: launch-post-1" \
  -H "Content-Type: application/json" \
  -d '{"brief":"Write a concise LinkedIn post announcing the launch."}'
```

A successful response contains only the completed editorial result:

```json
{"status":"completed","result":"The completed editorial copy."}
```

For `POST /brief`, authentication is checked before request framing, body reading,
validation, or queue submission. Accepted first-time jobs
use runtime source `EXTERNAL`, preserving the queue's FIFO and one-in-flight behavior.
Idempotency state is process-local and protected across concurrent requests: repeating the
same key and brief shares or returns the original response without another job, while using
the key with a different brief returns `409`. Accepted workflow failures are cached and are
not retried automatically. Queue rejections may be retried. All idempotency state resets
when the server restarts.

To demonstrate a cached repeat, run the `curl` command above twice without changing either
the key or body. Both calls return the same response, and only the first submits an external
runtime job. Placeholder tokens are examples only; never use them as real credentials.

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

### Heartbeat alert demo

`scripts/demo_heartbeat_notify.py` intentionally sends one real synthetic
repeated-failures alert through the normal AdminAgent, policy validation,
SQLite, renderer, and Telegram notifier flow. Normal heartbeat configuration
must be enabled and valid, and the additional explicit opt-in must normalize
exactly to `true`:

```shell
EDITORIAL_HEARTBEAT_DEMO_NOTIFY=true python scripts/demo_heartbeat_notify.py
```

Warning: this command makes a real model call and sends a real synthetic alert
to the configured maintainer Telegram chat. It does not start polling or the
automatic heartbeat scheduler.

## Structured-output reliability

The Coordinator and Critic request provider-native, JSON Schema-constrained output.
Their responses still pass through the application's strict JSON parser, domain
validation, and, for Critic issues, grounded-excerpt validation. The Talker, Writer,
and Editor continue to return plain text. Schema constraints improve formatting
reliability but do not make semantic agent output infallible.
