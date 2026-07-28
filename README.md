# Editorial Team v2

Provider-neutral infrastructure for a conversational editorial assistant.

This foundation contains model-client boundaries, a Gemini adapter, validation helpers,
sanitized errors, and a generic tool registry. Product behavior and orchestration are
intentionally out of scope at this stage.

## Development

Requires Python 3.11 or newer.

```shell
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Copy `.env.example` to `.env` and provide `GEMINI_API_KEY` to use the Gemini adapter.
