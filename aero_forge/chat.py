"""Interactive chat session for prompt-driven code generation and optimization."""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aero_forge.build_summary import format_build_summary
from aero_forge.bundle_repo import bundle_workspace, format_context_block
from aero_forge.config import ConfigOverride
from aero_forge.error_explainer import explain_error
from aero_forge.generate import (
    _find_generated_python_paths,
    extract_code_blocks,
    generate_and_build,
    optimize_generated_code,
)
from aero_forge.llm.clients import get_llm_client
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_RUST_PYTHON,
    classify_stack,
)
from aero_forge.scaffold.cargo_runner import run_cargo

logger = logging.getLogger("aero_forge.chat")

SESSION_DIR = Path.home() / ".cache" / "aero-forge" / "sessions"

ProgressCallback = Callable[[str], None]

COMMANDS = [
    "generate",
    "build",
    "test",
    "optimize",
    "faster",
    "benchmark",
    "show",
    "explain",
    "help",
    "exit",
    "quit",
]


_MULTIFILE_HINTS = {
    "rust core",
    "pyo3",
    "maturin",
    "cargo",
    "scripts/",
    "src/lib.rs",
    "tests/test_",
    "python engine",
    "python and rust",
    "rust/python",
    "hybrid",
    "polyglot",
}


def _is_multifile_hybrid_request(text: str) -> bool:
    """Return True when the prompt asks for a multi-file Rust/Python workspace."""
    lowered = text.lower()
    return any(hint in lowered for hint in _MULTIFILE_HINTS)


def _extract_file_path_from_block(code: str) -> Optional[str]:
    """Look for a ``// file: <path>`` or ``# file: <path>`` marker in the first lines."""
    for line in code.splitlines()[:5]:
        match = re.match(r"^\s*(?://|#)\s*file:\s*(\S+)", line)
        if match:
            return match.group(1).strip()
    return None


def _strip_file_marker(code: str) -> str:
    """Remove the ``// file:`` / ``# file:`` marker line from the start of a code block."""
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() and re.match(r"^\s*(?://|#)\s*file:\s*\S+", line):
            return "".join(lines[:i] + lines[i + 1 :]).strip("\n") + "\n"
    return code


def _parse_multifile_response(
    response: str,
    output_dir: Path,
) -> Dict[Path, str]:
    """Parse an LLM response containing marked code blocks into workspace files."""
    files: Dict[Path, str] = {}
    for _lang, code in extract_code_blocks(response):
        if not code.strip():
            continue
        rel_path = _extract_file_path_from_block(code)
        if not rel_path:
            continue
        if any(part == ".." for part in Path(rel_path).parts):
            continue
        target = (output_dir / rel_path).resolve()
        try:
            target.relative_to(output_dir.resolve())
        except ValueError:
            continue
        files[target] = _strip_file_marker(code)
    return files


def _run_pytest(workspace: Path) -> Tuple[int, str, str]:
    """Run pytest in *workspace* and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    result = subprocess.run(
        ["python", "-m", "pytest", "tests", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _run_simulation(workspace: Path, timeout: float = 120.0) -> Tuple[int, str, str]:
    """Run ``python scripts/run_simulation.py`` if it exists."""
    script = workspace / "scripts" / "run_simulation.py"
    if not script.is_file():
        return -1, "", f"{script} not found"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    result = subprocess.run(
        ["python", str(script)],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _session_path(session_id: str) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{session_id}.json"


def _noop(_msg: str) -> None:
    pass


class ChatSession:
    """Maintain multi-turn conversation state and dispatch code actions.

    The session keeps a history of user/assistant messages, the most recent
    build result, the generated source, and the original prompt.  It can be
    persisted to disk and resumed via ``session_id``.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        max_retries: int = 3,
        prompt_template: Optional[str] = None,
        session_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        config_override: Optional[ConfigOverride] = None,
    ):
        self.output_dir = Path(output_dir)
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.prompt_template = prompt_template
        self.session_id = session_id or self._new_session_id()
        self.progress_callback = progress_callback or _noop
        self.config_override = (
            config_override
            or ConfigOverride(
                llm_provider=llm_provider,
                model=model,
                api_key=api_key,
                max_retries=max_retries,
            )
        )

        self.messages: List[Dict[str, str]] = []
        self.last_error: Optional[str] = None
        self.last_prompt: Optional[str] = None
        self.last_source: Optional[str] = None
        self.last_summary: Optional[str] = None
        self.last_build_result: Optional[Dict[str, Any]] = None

        self.system_prompt = (
            "You are Aero-Forge, a fast, friendly coding co-pilot. "
            "Talk like a helpful teammate: casual, short, and punchy. "
            "Use dense sentences; avoid walls of text and raw JSON. "
            "When the backend emits deterministic build/test logs, translate them into "
            "lively narrative summaries with clear next steps. "
            "When asked to produce or modify code, return the full implementation "
            "in a single Python fenced code block and, if tests are requested, "
            "a second fenced code block for pytest tests."
        )

        self._load_session()

    @staticmethod
    def _new_session_id() -> str:
        return uuid.uuid4().hex[:8]

    def _progress(self, message: str) -> None:
        self.progress_callback(message)

    def _load_session(self) -> None:
        path = _session_path(self.session_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.messages = data.get("messages", [])
            self.last_prompt = data.get("last_prompt")
            self.last_source = data.get("last_source")
            self.last_summary = data.get("last_summary")
            self.last_error = data.get("last_error")
            self.last_build_result = data.get("last_build_result")
            loaded_output = data.get("output_dir")
            if loaded_output:
                self.output_dir = Path(loaded_output)
            self.llm_provider = data.get("llm_provider", self.llm_provider)
            self.model = data.get("model", self.model)
            self.prompt_template = data.get("prompt_template", self.prompt_template)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load session %s: %s", self.session_id, exc)

    def _save_session(self) -> None:
        path = _session_path(self.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "output_dir": str(self.output_dir.resolve()),
            "llm_provider": self.llm_provider,
            "model": self.model,
            "prompt_template": self.prompt_template,
            "messages": self.messages,
            "last_prompt": self.last_prompt,
            "last_source": self.last_source,
            "last_summary": self.last_summary,
            "last_error": self.last_error,
            "last_build_result": self.last_build_result,
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save session %s: %s", self.session_id, exc)

    def process(self, text: str) -> str:
        """Handle a user turn: try command dispatch, then conversational reply."""
        text = text.strip()
        if not text:
            return "Hi there! What would you like to build?"

        lowered = text.lower()
        if lowered in {"exit", "quit"}:
            return "Goodbye!"

        action = self.handle_command(text)
        if action is not None:
            return self._format_action_result(action, text)

        suggestion = self.suggest_command(text)
        if suggestion:
            return (
                f"I'm not sure about '{text}'. Did you mean '{suggestion}'? "
                "Type 'help' for a list of commands."
            )

        return self.reply(text)

    def reply(self, text: str) -> str:
        """Append user message, call the LLM, and return the assistant response."""
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.messages.append({"role": "user", "content": text})
        self._save_session()

        client = get_llm_client(
            self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            config_override=self.config_override,
        )
        if client is None:
            return (
                "No LLM provider is configured. Set AERO_FORGE_LLM_PROVIDER "
                "or pass --llm-provider."
            )

        response = client.generate(self.messages, temperature=0.2)
        if not response:
            return "Hmm, the LLM returned an empty response. Try rephrasing?"

        self.messages.append({"role": "assistant", "content": response})
        self._save_session()
        return response

    def handle_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect action verbs and optionally execute a build/optimize step."""
        lowered = text.lower().strip()

        if lowered in ("help", "?"):
            return self._help_action()

        # Multi-file or hybrid Rust/Python prompts are always generation requests,
        # even when phrased as "update" or "optimize".
        if _is_multifile_hybrid_request(text):
            return self._generate_action(text)

        # Long, structured prompts are generation requests even if they contain
        # words like "test" or "optimize" mid-sentence.
        word_count = len(text.split())
        if word_count > 25 or (text.count("\n") > 1 and word_count > 10):
            return self._generate_action(text)

        if any(
            phrase in lowered
            for phrase in (
                "generate",
                "write a function",
                "create",
                "implement",
                "build a",
            )
        ):
            return self._generate_action(text)

        if any(
            phrase in lowered
            for phrase in ("optimize", "make it faster", "speed up", "faster")
        ):
            return self._optimize_action(text)

        if "use less memory" in lowered or "less memory" in lowered:
            return self._optimize_action(text, constraints="Use less memory")

        if any(phrase in lowered for phrase in ("benchmark", "how fast", "time it")):
            return self._benchmark_action(text)

        if any(
            phrase in lowered for phrase in ("build", "compile", "run tests")
        ) or lowered in ("test", "run tests"):
            return self._build_action(text)

        if any(phrase in lowered for phrase in ("show", "display", "view code")):
            return self._show_action()

        if any(
            phrase in lowered
            for phrase in (
                "explain the algorithm",
                "explain the code",
                "how does it work",
            )
        ):
            return self._explain_algorithm_action()

        if any(phrase in lowered for phrase in ("explain", "why")):
            return self._explain_action()

        return None

    def suggest_command(self, text: str) -> Optional[str]:
        """Return the closest known command, or None if no good match."""
        lowered = text.lower().strip()
        if len(lowered) < 2:
            return None
        matches = difflib.get_close_matches(lowered, COMMANDS, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _format_action_result(self, result: Dict[str, Any], prompt: str) -> str:
        if "message" in result:
            return result["message"]

        if "iterations" in result:
            return self._summarize_iterations(result, prompt)

        if "build" in result:
            return self._summarize_build(result, prompt)

        return "Done!"

    def _has_source(self) -> bool:
        try:
            source_path, _ = _find_generated_python_paths(self.output_dir)
        except FileNotFoundError:
            return bool(self.last_source)
        return source_path.is_file() or bool(self.last_source)

    def _read_source(self) -> Optional[str]:
        source_path, _ = _find_generated_python_paths(self.output_dir)
        if source_path.is_file():
            return source_path.read_text(encoding="utf-8")
        return self.last_source

    def _generate_action(self, text: str) -> Dict[str, Any]:
        """Generate code from the user's prompt and build it."""
        self._progress("Sure! Generating code...")
        self.last_prompt = text
        classification = classify_stack(text)
        if classification.architecture in (
            INTENT_HYBRID_RUST_PYTHON,
            INTENT_HYBRID_CPP_PYTHON,
        ) or _is_multifile_hybrid_request(text):
            return self._multi_file_generate_action(text)
        result = generate_and_build(
            text,
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            max_iterations=self.max_iterations,
            prompt_template=self.prompt_template,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
            progress_callback=self._progress,
            config_override=self.config_override,
        )
        self._update_memory(text, result)
        return result

    def _multi_file_generate_action(self, text: str) -> Dict[str, Any]:
        """Generate a multi-file Rust/Python workspace from the prompt and run tests."""
        self._progress("Building multi-file hybrid workspace...")
        self.last_prompt = text
        self.output_dir.mkdir(parents=True, exist_ok=True)

        client = get_llm_client(
            self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            config_override=self.config_override,
        )
        if client is None:
            return {"message": "No LLM provider configured for multi-file generation."}

        system = (
            "You are a senior Rust and Python engineer. Given the user's request, "
            "produce a complete, buildable workspace. Return every file as a separate "
            "Markdown code fence. The first non-blank line inside each fence must be a "
            "comment containing the relative file path, using one of these forms:\n"
            "- Rust files: `// file: path/to/file.rs`\n"
            "- TOML files: `# file: path/to/Cargo.toml`\n"
            "- Python files: `# file: path/to/file.py`\n\n"
            "At minimum emit:\n"
            "- `Cargo.toml` (workspace root) and `rust_core/Cargo.toml` with a PyO3 cdylib crate\n"
            "- `rust_core/src/lib.rs` exposing the requested Rust functions through PyO3\n"
            "- `pyproject.toml` (optional package metadata)\n"
            "- `scripts/run_simulation.py` as a runnable entry point\n"
            "- `tests/test_metrics.py` pytest tests that import the generated module and assert behavior\n\n"
            "Use `pyo3 = { version = \"0.20.3\", features = [\"extension-module\", \"abi3-py39\", \"generate-import-lib\"] }`. "
            "Set crate-type = [\"cdylib\"] in the crate Cargo.toml. "
            "Ensure the compiled `.so` name matches the `[lib] name` in `rust_core/Cargo.toml` so Python can import it. "
            "Python wrapper code should look for the compiled shared object in the workspace root, `rust_core/target/release`, `target/release`, or `dist`.\n\n"
            "Do not include explanatory text outside the code fences.\n\n"
            "If CURRENT_PROJECT_CONTEXT is provided below, preserve existing file paths and "
            "behavior unless the user explicitly asks to change them." 
        )

        # Inject the current workspace state so follow-up turns know what already exists.
        bundle = bundle_workspace(self.output_dir, max_file_size_kb=50)
        if bundle["files"] or bundle["blueprint"]:
            system = system + "\n\n" + format_context_block(bundle, fmt="xml")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        try:
            response = client.generate(messages, temperature=0.2)
        except Exception as exc:
            return {"message": f"LLM call failed: {exc}"}

        if not response:
            return {"message": "LLM returned an empty response."}

        files = _parse_multifile_response(response, self.output_dir)
        if not files:
            return {
                "message": (
                    "I couldn't parse any marked files from the LLM response. "
                    "Try a more specific prompt."
                )
            }

        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._progress(f"Wrote {path.relative_to(self.output_dir)}")

        # Build the Rust crate if present.
        cargo_result: Optional[subprocess.CompletedProcess[str]] = None
        for cargo_toml in [self.output_dir / "Cargo.toml", self.output_dir / "rust_core" / "Cargo.toml"]:
            if cargo_toml.is_file():
                crate_dir = cargo_toml.parent
                self._progress(f"Compiling {cargo_toml.relative_to(self.output_dir)}...")
                cargo_result = run_cargo(
                    ["build", "--release"],
                    cwd=crate_dir,
                    retries=3,
                    timeout=600,
                )
                if cargo_result.returncode != 0:
                    return {
                        "message": (
                            f"Cargo build failed in {crate_dir.name}: "
                            f"{cargo_result.stderr or cargo_result.stdout}"
                        ),
                        "build": {"success": False, "error": cargo_result.stderr},
                    }
                # Make compiled shared objects importable from the workspace root.
                dist_dir = self.output_dir / "dist"
                dist_dir.mkdir(parents=True, exist_ok=True)
                search_roots = [
                    self.output_dir / "target" / "release",
                    self.output_dir / "rust_core" / "target" / "release",
                    crate_dir / "target" / "release",
                ]
                for search in search_roots:
                    for so in search.glob("*.so"):
                        # Python imports extension modules as <name>.so, not lib<name>.so.
                        module_so = so.name
                        if module_so.startswith("lib") and module_so.endswith(".so"):
                            module_so = module_so[3:]
                        destinations = [
                            dist_dir / so.name,
                            self.output_dir / so.name,
                            dist_dir / module_so,
                            self.output_dir / module_so,
                            search / module_so,
                        ]
                        for dst in destinations:
                            if so.resolve() == dst.resolve():
                                continue
                            shutil.copy(so, dst)
                break

        # Run pytest if tests exist.
        pytest_rc, pytest_stdout, pytest_stderr = _run_pytest(self.output_dir)
        tests_passed = pytest_rc == 0

        file_list = sorted(str(p.relative_to(self.output_dir)) for p in self.output_dir.rglob("*") if p.is_file())
        result = {
            "success": tests_passed,
            "build": {
                "success": (cargo_result is None or cargo_result.returncode == 0) and tests_passed,
                "cargo_returncode": cargo_result.returncode if cargo_result else 0,
                "pytest_returncode": pytest_rc,
                "pytest_output": pytest_stdout,
                "pytest_error": pytest_stderr,
            },
            "files": file_list,
        }
        if result["build"]["success"]:
            result["message"] = (
                f"Generated {len(files)} files and ran the full Rust/Python build. "
                f"Tests {'passed' if tests_passed else 'failed'}. "
                f"Output is in `{self.output_dir}`."
            )
        else:
            result["message"] = (
                f"Generated {len(files)} files, but the build did not pass all tests. "
                f"Check the build log for details."
            )
        self._update_memory(text, result)
        return result

    def _ensure_source_file(self) -> bool:
        """Write ``last_source`` to disk if the generated source file is missing."""
        source_path, _ = _find_generated_python_paths(self.output_dir)
        if not source_path.is_file() and self.last_source:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(self.last_source, encoding="utf-8")
        return source_path.is_file()

    def _build_action(self, text: str) -> Dict[str, Any]:
        """Compile the current generated project."""
        if not self._has_source():
            return {
                "message": "I don't have any code to build yet. Try 'generate <prompt>' first."
            }
        if not self._ensure_source_file():
            return {
                "message": "I have source in memory but couldn't write it. Try 'generate <prompt>' first."
            }
        self._progress("Got it, building now...")
        result = generate_and_build(
            self.last_prompt or "the function",
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            prompt_template=self.prompt_template,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
            progress_callback=self._progress,
            config_override=self.config_override,
        )
        self._update_memory(self.last_prompt or text, result)
        return result

    def _optimize_action(
        self, text: str, constraints: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize the current generated project."""
        if not self._has_source():
            return {
                "message": "I don't have any code to optimize yet. Try 'generate <prompt>' first."
            }
        if not self._ensure_source_file():
            return {
                "message": "I have source in memory but couldn't write it. Try 'generate <prompt>' first."
            }

        prompt = self.last_prompt or text
        self._progress("Alright, optimizing...")
        iterations = optimize_generated_code(
            output_dir=self.output_dir,
            prompt=prompt,
            constraints=constraints,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            max_iterations=self.max_iterations,
            prompt_template=self.prompt_template,
            progress_callback=self._progress,
            config_override=self.config_override,
        )
        self._progress("Optimization complete, rebuilding...")
        result = generate_and_build(
            prompt,
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            prompt_template=self.prompt_template,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
            progress_callback=self._progress,
            config_override=self.config_override,
        )
        result["iterations"] = iterations
        self._update_memory(prompt, result)
        return result

    def _benchmark_action(self, text: str) -> Dict[str, Any]:
        """Build and time the current generated project."""
        start = time.perf_counter()
        result = self._build_action(text)
        elapsed = time.perf_counter() - start
        result["benchmark_seconds"] = elapsed
        return result

    def _show_action(self) -> Dict[str, Any]:
        """Return the contents of the most recently generated source file."""
        source = self._read_source()
        if not source:
            return {"message": "No generated code yet. Try 'generate <prompt>' first."}
        return {"message": f"Here's the code:\n\n```python\n{source}\n```"}

    def _explain_algorithm_action(self) -> Dict[str, Any]:
        """Explain the current generated code in plain English."""
        source = self._read_source()
        if not source:
            return {"message": "No generated code yet. Build something first."}

        client = get_llm_client(
            self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            config_override=self.config_override,
        )
        if client is None:
            return {
                "message": (
                    "I have the source, but no LLM provider is configured to explain it. "
                    "Set AERO_FORGE_LLM_PROVIDER or pass --llm-provider."
                )
            }

        prompt = (
            "Explain this Python function in 2-3 punchy, casual sentences. "
            "Highlight the algorithm and one interesting tradeoff.\n\n"
            f"```python\n{source}\n```"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        explanation = client.generate(messages, temperature=0.3)
        if not explanation:
            return {"message": "I couldn't generate an explanation right now."}
        return {"message": f"Here's how it works:\n\n{explanation.strip()}"}

    def _explain_action(self) -> Dict[str, Any]:
        """Explain the last build error."""
        if not self.last_error:
            return {"message": "No error to explain. Build something first."}
        source = self._read_source()
        explanation = explain_error(
            self.last_error,
            source=source,
            llm_provider=self.llm_provider,
            model=self.model,
            config_override=self.config_override,
        )
        return {"message": f"Here's what went wrong:\n\n{explanation}"}

    def _help_action(self) -> Dict[str, Any]:
        """Return a friendly help message for chat commands."""
        return {
            "message": (
                "Here are some things you can say:\n"
                "  'Build a fast Fibonacci function' – generate and compile code\n"
                "  'Make it faster' / 'Use less memory' – optimize the current code\n"
                "  'Benchmark it' – build and time the project\n"
                "  'Show me the code' – display the generated source\n"
                "  'Explain the algorithm' – get a plain-English explanation\n"
                "  'Explain' – explain the last build error\n"
                "  'help' – show this message\n"
                "  'exit' or 'quit' – leave the chat\n\n"
                "You can also just chat naturally about what you'd like to build."
            )
        }

    def _update_memory(self, prompt: Optional[str], result: Dict[str, Any]) -> None:
        self.last_prompt = prompt or self.last_prompt
        self.last_build_result = result
        source = self._read_source()
        if source:
            self.last_source = source
        build = result.get("build") or {}
        if not build.get("success"):
            self.last_error = str(build.get("error", build))
        else:
            self.last_error = None
        self._save_session()

    def _summarize_build(self, result: Dict[str, Any], prompt: Optional[str]) -> str:
        build = result.get("build") or {}
        if not build.get("success"):
            error = build.get("error", "the build didn't succeed")
            self.last_error = str(error)
            return (
                f"Oops, I hit a snag: {error}\n\n"
                "You can type 'explain' for details, or try rephrasing your request."
            )

        function_names = self._function_names_from_result(result)

        summary = self._generate_build_summary(result, prompt, function_names)
        self.last_summary = summary
        self._save_session()
        return summary

    def _summarize_iterations(
        self, result: Dict[str, Any], prompt: Optional[str]
    ) -> str:
        iterations = result.get("iterations") or []
        if not iterations:
            return self._summarize_build(result, prompt)

        last = iterations[-1]
        build = last.get("build") or {}
        if not build.get("success"):
            return self._summarize_build(result, prompt)

        function_names = self._function_names_from_result(result)

        benchmark = last.get("benchmark_seconds")
        if benchmark is not None:
            extra = f" The last build took {benchmark:.3f}s."
        else:
            extra = ""

        summary = self._generate_build_summary(result, prompt, function_names)
        return f"{summary}{extra}"

    def _function_names_from_result(self, result: Dict[str, Any]) -> List[str]:
        build = result.get("build") or {}
        results = build.get("results") or []
        names: List[str] = []
        for r in results:
            name = r.get("function_name") or r.get("name")
            if name and name not in names:
                names.append(str(name))
        if not names and self.last_prompt:
            # Best-effort extraction from the generated source.
            source = self._read_source() or ""
            import ast as _ast

            try:
                tree = _ast.parse(source)
                for node in tree.body:
                    if isinstance(node, _ast.FunctionDef):
                        names.append(node.name)
            except Exception:
                pass
        return names

    def _generate_build_summary(
        self, result: Dict[str, Any], prompt: Optional[str], function_names: List[str]
    ) -> str:
        build = result.get("build") or {}

        benchmark = None
        if "iterations" in result and result["iterations"]:
            benchmark = result["iterations"][-1].get("benchmark_seconds")
        if benchmark is None:
            benchmark = result.get("benchmark_seconds")

        return format_build_summary(
            build,
            output_dir=self.output_dir / "dist",
            prompt=prompt,
            function_names=function_names,
            benchmark_seconds=benchmark,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            config_override=self.config_override,
        )


__all__ = ["ChatSession", "SESSION_DIR"]
