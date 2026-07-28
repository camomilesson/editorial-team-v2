# Editorial Team v2 — Agent Instructions

## Working scope

- Work only inside this repository.
- The old `Project` repository is read-only reference material.
- Never modify, move, delete, format, or commit files in the reference repository.
- Never introduce imports or runtime dependencies on the reference repository.

## Secrets and environment files

- Treat `.env`, `.env.*`, API keys, tokens, credentials, and private configuration
  as restricted secrets.
- Never open, read, search, print, summarize, quote, copy, diff, or modify `.env`
  files unless the user explicitly requests a specific operation.
- Use `.env.example` to learn environment-variable names and expected structure.
- Never reveal environment-variable values in output, logs, tests, exceptions,
  commits, or generated files.
- Never run commands that dump the environment, including `env`, `printenv`,
  `set`, or equivalent commands.
- Never commit `.env` or remove it from `.gitignore`.
- Tests must use fakes, mocks, or placeholder credentials.
- If a task appears to require access to a real secret, stop and ask the user.

## Commands and permissions

- Do not make network requests unless the task explicitly requires them.
- Do not push commits unless explicitly requested.
- Do not run destructive Git or filesystem commands.
- Review `git diff` and `git status` before reporting completion.

## Architecture

- This is a clean-slate conversational editorial assistant.
- Do not recreate legacy S3/S5 orchestration concepts or compatibility wrappers.
- Keep domain models independent of Gemini, Telegram, persistence, and prompts.
- Prefer small, explicit, provider-neutral interfaces.
- Preserve existing behavior and keep all tests passing unless the task explicitly
  changes an invariant.

## Verification

Before completing a coding task:

1. Run Ruff.
2. Run pytest.
3. Confirm `.env` and other secrets were not accessed or changed.
4. Confirm only intended repository files changed.