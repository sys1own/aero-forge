---
name: Testing the Aero-Forge copilot / builder API
scope: When testing aero-forge chat, blueprint synthesis, or the /api/build endpoint
description: How to run the manual E2E verification workflow for aero-forge PRs involving copilot responses, polyglot speedup actions, and trigger_build cards.
---

# Testing the Aero-Forge copilot / builder API

## Devin Secrets Needed
- `DEEPSEEK_API_KEY` (or `OPENROUTER_API_KEY` / `GEMINI_API_KEY` depending on the provider being tested)
- `AERO_FORGE_API_KEY` (fallback used by the LLM client if provider-specific key is absent)

## Environment assumptions
- Rust toolchain with `cargo` and `rustup` is installed.
- `aero-forge` repo is checked out on the branch under test, usually `main` after a PR is merged.
- Native `.so` extension is built:
  ```bash
  (cd aero_forge/_native && cargo build --release)
  cp aero_forge/_native/target/release/libaero_forge_native.so aero_forge/_native/aero_forge_native.so
  ```
- Python package is installed in editable mode:
  ```bash
  pip install -e ".[dev]"
  ```

## Command equivalents (current CLI is not exactly the one in the verification script)
- `aero-forge init-blueprint --path <dir>` is **not implemented**. Use:
  ```bash
  aero-forge blueprint synthesize --workspace <dir> --provider deepseek --output <dir>/blueprint.aero
  ```
- `aero-forge serve --port 8000` is **not implemented**. Use:
  ```bash
  aero-forge web --port 8000 --no-browser
  # or
  python -m aero_forge.server --port 8000 --no-browser
  ```

## API endpoints (current implementation)
- `POST /api/chat` — workspace-aware copilot chat.
  - If you use the `"message"` shorthand, the server **overwrites `chat.messages`** with only the user message, dropping the system prompt. This causes DeepSeek's JSON-mode request to fail and the model to fall back to plain text. To test the intended structured response, send the full `messages` array including the copilot system prompt.
- `POST /api/builder/trigger` — **does not exist**. It returns `404 {"error": "Not found"}`.
- `POST /api/build` — the actual trigger used by the web UI. It expects `prompt`, `provider`, `api_key`, and optionally `target_language` / `acceleration_policy`. It builds into a session sandbox (`~/.cache/aero-forge/sessions/` or `/tmp/aero-forge-sandboxes/<session_id>`), not directly into the supplied `workspace_path`.

## Typical validation curl

```bash
export DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY

curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"workspace_path\": \"/tmp/test_workspace\",
    \"provider\": \"deepseek\",
    \"api_key\": \"$DEEPSEEK_API_KEY\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"<copilot system prompt>\"},
      {\"role\": \"user\", \"content\": \"Analyze blueprint.aero and recommend a high-performance C++ or Rust extension to accelerate array operations and matrix computations in this project.\"}
    ]
  }"
```

Then to build:

```bash
curl -s -X POST http://localhost:8000/api/build \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"<session_id from chat>\",
    \"prompt\": \"<clean_prompt from chat action>\",
    \"provider\": \"deepseek\",
    \"api_key\": \"$DEEPSEEK_API_KEY\",
    \"target_language\": \"hybrid_rust_python\",
    \"acceleration_policy\": \"selective\"
  }"
```

## What to check after a build
- The response `status` should be `success`.
- `result.build.success` should be `true`.
- `result.build.test_passed` should equal `result.build.test_total`.
- A compiled `.so` must exist in the session sandbox (`target/release/*.so` or `rust_core/target/release/*.so`).
- If the build "skips Rust crate build" with `No accelerable contracts found`, the blueprint's ABI contracts likely contain `memory_model` values the validator rejects, or the signatures use unsupported C-pointer syntax. Inspect `/tmp/aero-forge-sandboxes/<session>/blueprint.aero`.

## Common failure signatures
- `"Prompt must contain the word 'json' ... 'json_object'"` in server logs → the copilot system prompt was not sent to the LLM (message-list overwrite bug).
- `Intent JSON schema validation failed: memory_model must be one of ...` → the LLM emitted an invalid `memory_model` value in the generated blueprint.
- `generate_monorepo failed; falling back to PolyglotMaterializer` then `No accelerable contracts found` → the materializer cannot match the generated contracts to an accelerable Rust function and skips the crate build.
- `matmul` returns `None` in tests → the `.so` was never built or the Python wrapper could not load it.
