---
name: Testing the Aero-Forge copilot / builder API
scope: When testing aero-forge chat, blueprint synthesis, /api/build, or /api/builder/trigger
description: How to run the manual E2E verification workflow for aero-forge PRs involving copilot responses, polyglot speedup actions, and trigger_build cards.
---

# Testing the Aero-Forge copilot / builder API

## Devin Secrets Needed
- `DEEPSEEK_API_KEY` (or `OPENROUTER_API_KEY` / `GEMINI_API_KEY` depending on the provider being tested)
- `AERO_FORGE_API_KEY` (fallback used by the LLM client if provider-specific key is absent)

## Environment assumptions
- Rust toolchain with `cargo` and `rustup` is installed.
- `aero-forge` repo is checked out on the branch under test.
- Native `.so` extension is built:
  ```bash
  (cd aero_forge/_native && cargo build --release)
  cp aero_forge/_native/target/release/libaero_forge_native.so aero_forge/_native/aero_forge_native.so
  ```
- Python package is installed in editable mode:
  ```bash
  pip install -e ".[dev]"
  ```

## CLI commands
- `aero-forge init-blueprint --path <dir>` works on recent branches. It auto-detects the workspace and writes `blueprint.aero`, reusing an existing blueprint if present.
- `aero-forge serve --port 8000 --no-browser` is an alias for `aero-forge web`, but some branches have a click wiring bug that causes `TypeError: Context.__init__() got an unexpected keyword argument 'port'`. If `serve` fails, fall back to `aero-forge web --port 8000 --no-browser`.
- `aero-forge web --port 8000 --no-browser` reliably starts the aiohttp server on port 8000.

## API endpoints
- `POST /api/chat` — workspace-aware copilot chat.
  - Sending only `"message"` should now preserve the copilot system prompt and return structured JSON with `display_text` containing an Architecture Overview and a build action.
  - The response shape includes `action` (canonical, may have `type: "build"`), `clean_prompt`, and `legacy_action` (often `type: "SUGGEST_BUILD_PROMPT"`).
- `POST /api/builder/trigger` — alias for `POST /api/build` on recent branches.
  - Accepts `builder_prompt` (and also `prompt`) plus `workspace_path`, `provider`, `api_key`, `target_language`, `acceleration_policy`, `architecture`, and `session_id`.
  - When `workspace_path` is supplied, the builder uses it as the output directory and seeds contracts from `<workspace_path>/blueprint.aero`.
- `POST /api/build` — the original trigger, still used by the web UI.

## Typical validation curl flow
```bash
export DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY

curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"workspace_path\": \"/tmp/test_workspace\",
    \"provider\": \"deepseek\",
    \"api_key\": \"$DEEPSEEK_API_KEY\",
    \"message\": \"Analyze blueprint.aero and recommend a high-performance C++ or Rust extension to accelerate array operations and matrix computations in this project.\"
  }" | tee /tmp/chat_response.json
```

Then build:
```bash
BUILDER_PROMPT=$(jq -r '.clean_prompt // .action.clean_prompt // empty' /tmp/chat_response.json)
TARGET=$(jq -r '.action.parameters.target // .legacy_action.params.target // "auto"' /tmp/chat_response.json)
ACCEL=$(jq -r '.action.parameters.acceleration // .legacy_action.params.acceleration // "selective"' /tmp/chat_response.json)
ARCH=$(jq -r '.action.parameters.architecture // .legacy_action.params.parameters.architecture // empty' /tmp/chat_response.json)

curl -s -X POST http://localhost:8000/api/builder/trigger \
  -H "Content-Type: application/json" \
  -d "{
    \"workspace_path\": \"/tmp/test_workspace\",
    \"provider\": \"deepseek\",
    \"api_key\": \"$DEEPSEEK_API_KEY\",
    \"builder_prompt\": \"$BUILDER_PROMPT\",
    \"target_language\": \"$TARGET\",
    \"acceleration_policy\": \"$ACCEL\",
    \"architecture\": \"$ARCH\"
  }" | tee /tmp/builder_trigger_response.json
```

## What to check after a build
- HTTP `200` and `result.build.success == true`.
- `result.build.test_passed == result.build.test_total`.
- A compiled native artifact (`*.so`, `*.pyd`, or `*.dylib`) exists in the build directory.
- `pytest` passes in the workspace.

## Common failure signatures
- `TypeError: Context.__init__() got an unexpected keyword argument 'port'` when running `aero-forge serve` → the CLI alias is calling a click `Command` as a plain function.
- `"Prompt must contain the word 'json' ... 'json_object'"` in server logs → the copilot system prompt was dropped before the LLM call.
- `Intent JSON schema validation failed: memory_model must be one of ...` or `binding_framework must be one of ...` → the LLM emitted blueprint contract values that the schema does not accept.
- `generate_monorepo failed; falling back to PolyglotMaterializer` followed by `Missing declared file src/native_ops/Cargo.toml from blueprint.aero` (or similar) → the planner's `module_graph` and the materializer's generated file paths disagree.
- `assert None is not None` in generated tests → the Rust functions are stubs returning `()` and the Python wrapper returns `None` for all calls, including functions the tests expect to return a value.
- `ModuleNotFoundError: No module named 'accelerator'` when running workspace tests → the workspace's package is not installed in editable mode (`pip install -e /tmp/test_workspace`) before `pytest`.
