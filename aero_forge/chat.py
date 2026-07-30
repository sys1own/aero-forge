"""Interactive chat session for prompt-driven code generation and optimization."""

from __future__ import annotations

import ast
import difflib
import json
import logging
import os
import re

try:
    import tomllib as toml_reader
except ImportError:  # pragma: no cover
    try:
        import tomli as toml_reader
    except ImportError:  # pragma: no cover
        toml_reader = None  # type: ignore
import shutil
import subprocess
import time
import uuid
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aero_forge.build_summary import format_build_summary
from aero_forge.builder.executor import ExecutionReport
from aero_forge.bundle_repo import bundle_to_xml, bundle_workspace, format_context_block
from aero_forge.config import ConfigOverride, Tier
from aero_forge.context_bundler import ContextBundler, get_blueprint_status
from aero_forge.copilot.action_parser import (
    ActionParser,
    _has_build_intent,
    parse_action_from_text,
    sanitize_builder_prompt,
)
from aero_forge.copilot.agent import (
    _has_markdown_heading,
    _legacy_action_type,
    format_copilot_response,
    workspace_blueprint_tag,
)
from aero_forge.prompts import AERO_FORGE_COPILOT_SYSTEM_PROMPT
from aero_forge.error_explainer import explain_error
from aero_forge.healing.router import try_auto_fix
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
from aero_forge.scaffold.cli_normalizer import normalize_workspace
from aero_forge.scaffold.rust_merge import fix_rust_core_impls
from aero_forge.scaffold.syntax_guard import repair_workspace

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


def _ensure_root_cargo_workspace(workspace_dir: Path) -> bool:
    """Make sure the workspace root ``Cargo.toml`` is a valid Cargo workspace.

    LLMs occasionally emit a root ``Cargo.toml`` that references the ``rust_core``
    crate but omits the ``[workspace]`` section, causing ``cargo build`` to fail
    with "manifest is missing either a [package] or a [workspace]".  This function
    rewrites the root manifest as a minimal virtual workspace when it is invalid.
    """
    root_cargo = workspace_dir / "Cargo.toml"
    if not root_cargo.is_file():
        return False

    try:
        text = root_cargo.read_text(encoding="utf-8")
    except Exception:
        return False

    if toml_reader is not None:
        try:
            with root_cargo.open("rb") as f:
                parsed = toml_reader.load(f)
        except Exception:
            parsed = {}
        if parsed.get("workspace") or parsed.get("package"):
            return False
        if parsed.get("dependencies") or parsed.get("lib") or parsed.get("bin"):
            # This looks like a crate manifest without a package section; it
            # cannot serve as a workspace root.  Ignore it and build the crate directly.
            root_cargo.unlink()
            return True

    # Fallback: crude textual check for a manifest section.
    if re.search(r"^\s*\[(?:workspace|package)\]", text, re.MULTILINE):
        return False

    root_cargo.write_text(
        '[workspace]\nmembers = ["rust_core"]\nresolver = "2"\n\n' + text,
        encoding="utf-8",
    )
    return True


def _make_script_runnable(script_path: Path, workspace_dir: Path) -> bool:
    """Prepend ``sys.path`` setup to *script_path* when it imports a local package.

    Generated scripts in ``scripts/`` frequently import a package directory that
    lives at the workspace root.  When the script is run directly the root is not
    on ``sys.path``, so insert it before the first non-shebang line.
    """
    if not script_path.is_file():
        return False
    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return False
    if "sys.path.insert" in text:
        return False

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False

    top_level_imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node.names
    }
    local_packages = {
        name for name in top_level_imports
        if (workspace_dir / name).is_dir() and (workspace_dir / name / "__init__.py").is_file()
    }
    if not local_packages:
        return False

    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    block = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "\n"
    )
    new_text = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
    script_path.write_text(new_text, encoding="utf-8")
    return True


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


def get_session_metadata(session_id: str) -> Dict[str, Any]:
    """Return blueprint provenance and runtime metadata for a session."""
    path = _session_path(session_id)
    if not path.exists():
        return {"blueprint_source": "unknown", "auto_initialized": False, "is_building": False, "is_synthesizing": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"blueprint_source": "unknown", "auto_initialized": False, "is_building": False, "is_synthesizing": False}
    data.setdefault("blueprint_source", "unknown")
    data.setdefault("auto_initialized", False)
    data.setdefault("is_building", False)
    data.setdefault("is_synthesizing", False)
    return data


def set_session_blueprint_metadata(
    session_id: str,
    source: Optional[str] = None,
    auto_initialized: Optional[bool] = None,
    is_building: Optional[bool] = None,
    is_synthesizing: Optional[bool] = None,
) -> None:
    """Update blueprint provenance and runtime metadata for a session."""
    path = _session_path(session_id)
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    if source is not None:
        data["blueprint_source"] = source
    if auto_initialized is not None:
        data["auto_initialized"] = auto_initialized
    if is_building is not None:
        data["is_building"] = is_building
    if is_synthesizing is not None:
        data["is_synthesizing"] = is_synthesizing
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not update session metadata %s: %s", session_id, exc)


def _noop(_msg: str) -> None:
    pass


class WorkspaceContextHarvester:
    """Load a workspace and its ``blueprint.aero`` into an LLM-friendly context block."""

    def __init__(self, workspace_dir: Path, max_file_size_kb: int = 50):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.max_file_size_kb = max_file_size_kb
        self.blueprint_path = self.workspace_dir / "blueprint.aero"

    def _blueprint(self) -> Optional[Dict[str, Any]]:
        if not self.blueprint_path.is_file():
            return None
        try:
            text = self.blueprint_path.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}
        except Exception as exc:
            logger.warning("Could not parse blueprint %s: %s", self.blueprint_path, exc)
            return None

    def _source_files(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        max_bytes = self.max_file_size_kb * 1024
        skip_names = {
            "target",
            "dist",
            "build",
            "__pycache__",
            ".pytest_cache",
            ".aero",
            ".venv",
            ".git",
        }
        for path in sorted(self.workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if any(part in skip_names for part in rel.parts[:1]):
                continue
            if rel.name in ("blueprint.aero",):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
                files[rel.as_posix()] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        return files

    def harvest(self) -> Dict[str, Any]:
        blueprint = self._blueprint() or {}
        llm_context = blueprint.get("llm_context") or {}
        metadata = blueprint.get("metadata") or {}
        return {
            "workspace": str(self.workspace_dir),
            "blueprint": blueprint,
            "llm_initialized": bool(
                llm_context.get("state") == "synthesized"
                or metadata.get("generation_method") == "llm_synthesized"
                or metadata.get("llm_initialized")
            ),
            "llm_context": {
                "state": llm_context.get("state", "raw"),
                "repository_summary": llm_context.get("repository_summary", ""),
                "dependency_graph": llm_context.get("dependency_graph", {}),
                "exported_api_signatures": llm_context.get("exported_api_signatures", {}),
                "polyglot_boundaries": llm_context.get("polyglot_boundaries", []),
                "compute_hotspots": llm_context.get("compute_hotspots", []),
            },
            "source_files": self._source_files(),
        }

    def to_prompt(self) -> str:
        data = self.harvest()
        blueprint_text = None
        if data["blueprint"]:
            blueprint_text = yaml.safe_dump(data["blueprint"], sort_keys=False)
        bundle = {
            "workspace": data["workspace"],
            "files": data["source_files"],
            "blueprint": blueprint_text,
            "test_status": None,
        }
        body = bundle_to_xml(bundle)
        ctx = data["llm_context"]
        extra: List[str] = []
        if ctx.get("repository_summary"):
            extra.append(f"Repository summary: {ctx['repository_summary']}")
        if ctx.get("dependency_graph"):
            extra.append("Dependency graph:")
            for src, deps in ctx["dependency_graph"].items():
                extra.append(f"  {src} -> {', '.join(deps) if deps else 'none'}")
        if ctx.get("exported_api_signatures"):
            extra.append("Exported API signatures:")
            for src, signatures in ctx["exported_api_signatures"].items():
                extra.append(f"  {src}:")
                for sig in signatures:
                    extra.append(f"    - {sig}")
        if ctx.get("polyglot_boundaries"):
            extra.append("Polyglot boundaries:")
            for boundary in ctx["polyglot_boundaries"]:
                extra.append(f"  {boundary}")
        if ctx.get("compute_hotspots"):
            extra.append("Compute hotspots:")
            for hotspot in ctx["compute_hotspots"]:
                name = hotspot.get("name", "unknown")
                file = hotspot.get("file", "")
                complexity = hotspot.get("complexity", "")
                reason = hotspot.get("reason", "")
                candidate = hotspot.get("acceleration_candidate", True)
                extra.append(
                    f"  {name} ({file}) complexity={complexity} acceleration={candidate}"
                )
                if reason:
                    extra.append(f"    reason: {reason}")
        if extra:
            body = "\n".join(["[LLM_CONTEXT]"] + extra) + "\n" + body
        return f"CURRENT_PROJECT_CONTEXT (XML workspace bundle):\n{body}\n---END PROJECT CONTEXT---"


class ChatSession:
    """Maintain multi-turn conversation state and dispatch code actions.

    The session keeps a history of user/assistant messages, the most recent
    build result, the generated source, and the original prompt.  It can be
    persisted to disk and resumed via ``session_id``.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        workspace_path: Optional[Path] = None,
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
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif workspace_path is not None:
            self.output_dir = Path(workspace_path)
        else:
            self.output_dir = Path(".").resolve()
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
        self.project_context: Optional[str] = None
        self.last_error_context: Optional[str] = None

        # Terminal context for self-healing
        self.last_terminal_command: Optional[str] = None
        self.last_terminal_exit_code: Optional[int] = None
        self.last_terminal_log: Optional[str] = None
        self.last_evaluator_state: Optional[Dict[str, Any]] = None

        # Blueprint provenance for the web workspace
        self.blueprint_source: str = "unknown"
        self.auto_initialized: bool = False

        self.base_system_prompt = (
            "You are Aero-Forge, a fast, friendly coding co-pilot. "
            "Talk like a helpful teammate: casual, short, and punchy. "
            "Use dense sentences; avoid walls of text and raw JSON. "
            "When the backend emits deterministic build/test logs, translate them into "
            "lively narrative summaries with clear next steps. "
            "When asked to produce or modify code, return the full implementation "
            "in a single Python fenced code block and, if tests are requested, "
            "a second fenced code block for pytest tests."
        )
        self.copilot_system_prompt = AERO_FORGE_COPILOT_SYSTEM_PROMPT
        self.system_prompt = self.base_system_prompt

        requested_output_dir = self.output_dir
        self._load_session()
        # A loaded session may point to a temp directory that no longer exists;
        # fall back to the directory supplied by the caller.
        if not self.output_dir.is_dir():
            self.output_dir = requested_output_dir
        self._refresh_project_context()

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
            self.project_context = data.get("project_context")
            self.last_error_context = data.get("last_error_context")
            self.last_terminal_command = data.get("last_terminal_command")
            self.last_terminal_exit_code = data.get("last_terminal_exit_code")
            self.last_terminal_log = data.get("last_terminal_log")
            self.last_evaluator_state = data.get("last_evaluator_state")
            self.blueprint_source = data.get("blueprint_source", self.blueprint_source)
            self.auto_initialized = data.get("auto_initialized", self.auto_initialized)
            loaded_output = data.get("output_dir")
            if loaded_output:
                self.output_dir = Path(loaded_output)
            self.llm_provider = data.get("llm_provider", self.llm_provider)
            self.model = data.get("model", self.model)
            self.prompt_template = data.get("prompt_template", self.prompt_template)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load session %s: %s", self.session_id, exc)

    def _refresh_project_context(self) -> None:
        """Build and cache a compact workspace bundle for the LLM context."""
        self.system_prompt = self.base_system_prompt
        self.project_context = None
        if not self.output_dir.is_dir():
            return
        try:
            context = WorkspaceContextHarvester(self.output_dir, max_file_size_kb=50).to_prompt()
        except Exception as exc:
            logger.warning("Could not harvest workspace %s: %s", self.output_dir, exc)
            return
        if context:
            self.project_context = context
            self.system_prompt = self.base_system_prompt + "\n\n" + context

        if self.last_terminal_log:
            terminal_summary = (
                f"[TERMINAL CONTEXT]\nCommand: {self.last_terminal_command}\n"
                f"Exit code: {self.last_terminal_exit_code}\n"
                f"LogEvaluator: {self.last_evaluator_state}\n"
            )
            self.system_prompt += "\n\n" + terminal_summary

    def _copilot_system_prompt(self) -> str:
        """Build the workspace-aware copilot system prompt used by the web UI."""
        prompt = self.copilot_system_prompt
        status = get_blueprint_status(self.output_dir)
        if status["llm_initialized"]:
            workspace_status = (
                "Workspace has an initialized blueprint; focus on update/refactor planning that "
                "integrates with the existing architecture."
            )
        elif status["exists"]:
            workspace_status = (
                "Workspace blueprint exists but is not LLM-initialized; treat the current "
                "files as a preliminary snapshot and focus on drafting an initial build prompt."
            )
        elif status["source_count"]:
            workspace_status = (
                "Workspace has source files but no blueprint; treat the current files as a "
                "preliminary snapshot and focus on drafting an initial build prompt."
            )
        else:
            workspace_status = (
                "Workspace is empty; focus on understanding the user's intent and "
                "drafting an initial build prompt with target and contracts."
            )
        prompt += f"\n\n[WORKSPACE STATUS]\n{workspace_status}"
        if self.project_context:
            prompt += "\n\n" + self.project_context
        blueprint_tag = workspace_blueprint_tag(self.output_dir)
        if blueprint_tag:
            prompt += "\n\n" + blueprint_tag
        if self.last_terminal_log:
            prompt += (
                "\n\n[TERMINAL CONTEXT]\n"
                f"Command: {self.last_terminal_command}\n"
                f"Exit code: {self.last_terminal_exit_code}\n"
                f"LogEvaluator: {self.last_evaluator_state}\n"
            )
        if self.last_error_context:
            prompt += "\n\n[ERROR CONTEXT]\n" + self.last_error_context
        return prompt

    def set_terminal_context(
        self,
        command: str,
        exit_code: int,
        log_text: str,
    ) -> None:
        """Store the latest terminal failure and its LogEvaluator diagnosis."""
        from aero_forge.healing.evaluator import LogEvaluator

        self.last_terminal_command = command
        self.last_terminal_exit_code = exit_code
        self.last_terminal_log = log_text
        self.last_evaluator_state = LogEvaluator().evaluate_log(command, exit_code, log_text)

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
            "project_context": self.project_context,
            "last_error_context": self.last_error_context,
            "last_terminal_command": self.last_terminal_command,
            "last_terminal_exit_code": self.last_terminal_exit_code,
            "last_terminal_log": self.last_terminal_log,
            "last_evaluator_state": self.last_evaluator_state,
            "blueprint_source": self.blueprint_source,
            "auto_initialized": self.auto_initialized,
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

    def _maybe_parse_json_reply(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from an LLM response, accepting fenced blocks.

        Handles strict JSON, Markdown fences, a leading JSON object followed by
        prose, and a JSON object embedded anywhere in the text.
        """
        if not response or not response.strip():
            return None
        text = response.strip()

        # Strip Markdown code fences if present.
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try strict JSON first.
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        # Fall back to extracting the first balanced JSON object.
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                parsed = json.loads(text[start:end])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _infer_build_action(text: str) -> Optional[Dict[str, Any]]:
        """Construct a PROPOSE_BUILD action from prose if build keywords are present."""
        lowered = text.lower()
        build_keywords = (
            "build",
            "create",
            "implement",
            "generate",
            "function",
            "module",
            "project",
            "fast",
            "speed",
            "accelerate",
            "optimize",
        )
        if not any(k in lowered for k in build_keywords):
            return None

        has_python = "python" in lowered or "pyo3" in lowered
        has_rust = "rust" in lowered or "cargo" in lowered or "pyo3" in lowered
        has_cpp = "cpp" in lowered or "c++" in lowered or "cxx" in lowered

        if has_python and has_rust and has_cpp:
            target = "tri_polyglot_rust_cpp_python"
        elif has_python and has_rust:
            target = "hybrid_rust_python"
        elif has_python and has_cpp:
            target = "hybrid_cpp_python"
        elif has_rust and has_cpp:
            target = "hybrid_cpp_rust"
        elif has_rust:
            target = "pure_rust"
        elif has_cpp:
            target = "hybrid_cpp_python"
        else:
            target = "pure_python"

        if has_rust or has_cpp or "accelerate" in lowered or "fast" in lowered or "speed" in lowered:
            acceleration = "Selective Acceleration (Auto-Detect Heavy Compute)"
        else:
            acceleration = "Standard Runtime (Bypass Bridge)"

        prompt = text.strip()
        # Prefer to use the original user request if the response is too terse.
        if len(prompt) < 20 or prompt.startswith("{"):
            prompt = f"Build {text.strip()[:200]}"

        return {
            "type": "PROPOSE_BUILD",
            "params": {
                "prompt": prompt,
                "target": target,
                "acceleration": acceleration,
            },
        }

    def reply(self, text: str) -> str:
        """Append user message, call the LLM, and return the assistant response text."""
        structured = self.reply_structured(text)
        return structured["reply"]

    def reply_structured(
        self,
        text: str,
        error_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append user message, call the LLM, and return a structured reply + action.

        If ``error_context`` is provided, it is injected into the system prompt for
        this turn only and does not persist to the session file.
        """
        self.last_error_context = error_context
        self._refresh_project_context()

        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": self._copilot_system_prompt()})
        if not self.messages or self.messages[-1].get("role") != "user" or self.messages[-1].get("content") != text:
            self.messages.append({"role": "user", "content": text})
        self._save_session()

        client = get_llm_client(
            self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            config_override=self.config_override,
            tier=Tier.FAST,
        )
        if client is None:
            return {
                "reply": (
                    "No LLM provider is configured. Set AERO_FORGE_LLM_PROVIDER "
                    "or pass --llm-provider."
                ),
                "action": None,
                "explanation": "No LLM provider is configured.",
                "has_suggestion": False,
                "build_prompt": None,
                "raw": "",
            }

        # Try JSON mode first. Some providers return an empty body when forced
        # into JSON mode, so fall back to a plain generation attempt.
        try:
            response = client.generate(
                self.messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = client.generate(self.messages, temperature=0.2)

        if not response:
            logger.warning("LLM returned empty response in JSON mode; retrying plain.")
            try:
                response = client.generate(self.messages, temperature=0.2)
            except TypeError:
                response = ""

        if not response:
            fallback_action = self._fallback_build_action(text) if _has_build_intent(text) else None
            fallback = self._build_result(
                "I didn't receive a response from the language model. Please check your provider/API key and try rephrasing.",
                action=fallback_action,
                raw="",
            )
            self.messages.append({"role": "assistant", "content": json.dumps(fallback)})
            self._save_session()
            return fallback

        parsed = ActionParser().parse(response)
        display_text = parsed.get("display_text", "")
        action = parsed.get("action")

        # If the model returned nothing at all, recover a build action from the user text.
        if not response.strip() and not action and _has_build_intent(text):
            action = self._fallback_build_action(text)

        result = self._build_result(display_text, action=action, raw=response)
        self.messages.append({"role": "assistant", "content": result["display_text"] or response})
        self._save_session()
        return result

    def _build_result(self, display_text: str, action: Optional[Dict[str, Any]], raw: str) -> Dict[str, Any]:
        """Build a structured chat result compatible with both old and new UI fields."""
        clean_prompt = sanitize_builder_prompt(action.get("clean_prompt", "")) if action else None
        parameters = action.get("parameters") if action else {}
        legacy_action = None
        if action:
            legacy_action = {
                "type": _legacy_action_type(action),
                "params": {
                    "prompt": clean_prompt,
                    "target": parameters.get("target", "pure_python"),
                    "acceleration": parameters.get("acceleration", "Selective Acceleration (Auto-Detect Heavy Compute)"),
                    "parameters": parameters,
                },
            }
            # If the model left the conversational text empty, provide a concise
            # rationale rather than echoing the executable prompt.
            if not display_text:
                target_label = parameters.get("target", "pure_python").replace("_", " ").title()
                display_text = f"I propose a **{target_label}** build. Use the Action Card to edit or trigger it."
        # Wrap a heading around the conversational text when an action is attached
        # and the model did not supply one, but never hide a plain informational reply.
        if action and display_text and not _has_markdown_heading(display_text):
            display_text = "### Architecture Overview\n\n" + display_text

        return {
            "type": "chat_message",
            "display_text": display_text,
            "reply": display_text,
            "message": display_text,
            "action": legacy_action,
            "clean_action": action,
            "explanation": display_text,
            "has_suggestion": bool(action),
            "has_prompt": bool(clean_prompt),
            "build_prompt": clean_prompt,
            "suggested_build_prompt": clean_prompt,
            "suggested_prompt": clean_prompt,
            "clean_prompt": clean_prompt,
            "parameters": parameters,
            "raw": raw,
        }

    def _fallback_build_action(self, text: str) -> Dict[str, Any]:
        """Create a sanitized build action from plain user text when the LLM fails."""
        parsed = ActionParser().parse(text)
        action = parsed.get("action") or {
            "type": "build",
            "source": "plain_text",
            "clean_prompt": text.strip(),
            "parameters": {"target": "pure_python", "acceleration": "Selective Acceleration (Auto-Detect Heavy Compute)"},
            "blueprint": None,
        }
        action["clean_prompt"] = sanitize_builder_prompt(action.get("clean_prompt", ""))
        return action

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

        if any(
            phrase in lowered
            for phrase in (
                "fix error",
                "fix build",
                "apply self heal",
                "self heal",
                "heal",
            )
        ):
            return self._heal_action()

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

    def _build_and_test_workspace(self) -> Tuple[Optional[subprocess.CompletedProcess[str]], int, str, str, bool]:
        """Run post-processing, compile the Rust crate, and execute pytest.

        Returns ``(cargo_result, pytest_rc, pytest_stdout, pytest_stderr, tests_passed)``.
        """
        # Make generated runnable scripts import the workspace root package(s).
        scripts_dir = self.output_dir / "scripts"
        if scripts_dir.is_dir():
            for script in scripts_dir.glob("*.py"):
                if _make_script_runnable(script, self.output_dir):
                    self._progress(f"Made {script.relative_to(self.output_dir)} runnable from any cwd")

        # Repair common LLM truncation mistakes before compilation.
        for repaired in repair_workspace(self.output_dir):
            self._progress(f"Repaired syntax truncation in {repaired.relative_to(self.output_dir)}")

        # LLMs often emit a #[pymethods] impl whose methods duplicate an
        # inherent impl for the same struct; merge them before compiling.
        if fix_rust_core_impls(self.output_dir):
            self._progress("Merged duplicate Rust impl blocks in rust_core/src/lib.rs")

        # Guard against an LLM-generated root Cargo.toml that is missing
        # the [workspace] / [package] section.
        if _ensure_root_cargo_workspace(self.output_dir):
            self._progress("Fixed workspace root Cargo.toml")

        # Ensure LLM-generated CLI modules re-export native functions and that
        # __init__.py exposes public names from native.py / cli.py.
        for modified in normalize_workspace(self.output_dir):
            self._progress(f"Normalized exports in {modified}")

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
                    return cargo_result, 0, "", cargo_result.stderr or cargo_result.stdout, False
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

        # Run pytest if tests exist; attempt deterministic self-healing on
        # common test typos (e.g. ``r_stats`` when ``rstats`` is defined).
        pytest_rc, pytest_stdout, pytest_stderr = _run_pytest(self.output_dir)
        tests_passed = pytest_rc == 0
        for _ in range(2):
            if tests_passed:
                break
            combined = f"{pytest_stderr}\n{pytest_stdout}"
            name_error = re.search(r"NameError: name ['\"](\w+)['\"] is not defined", combined)
            file_line = re.search(r"([\w/]+\.py):\d+:", combined)
            if not name_error or not file_line:
                break
            bad_name = name_error.group(1)
            test_file = self.output_dir / file_line.group(1)
            if test_file.is_file():
                original = test_file.read_text(encoding="utf-8")
                fixed = try_auto_fix(combined, original)
                if fixed and fixed != original:
                    test_file.write_text(fixed, encoding="utf-8")
                    self._progress(f"Self-healed test typo: {bad_name} -> fixed in {test_file.name}")
                    pytest_rc, pytest_stdout, pytest_stderr = _run_pytest(self.output_dir)
                    tests_passed = pytest_rc == 0
                    continue
            break

        return cargo_result, pytest_rc, pytest_stdout, pytest_stderr, tests_passed

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
            tier=Tier.REASONING,
        )
        if client is None:
            return {"message": "No LLM provider configured for multi-file generation."}

        generation_rules = (
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
            "Test rules: for statistical / anomaly functions, generate tests that compare the compiled "
            "native output with the pure-Python fallback (parity) and use `math.isclose` / `pytest.approx` "
            "for floats. For anomaly detection, use a large sample with one extreme outlier (e.g. ``[1.0] * 100 + [10000.0]``) "
            "so the anomaly is reliably above a 3.0 sigma threshold. Avoid testing with an all-zero baseline, "
            "because zero standard deviation makes every non-zero value an anomaly; use a baseline with non-zero "
            "variance (e.g. ``[float(i) for i in range(20)]``) and compare the spike value against the threshold. "
            "Avoid brittle exact scalar equality like ``assert res['anomalies'] == 1``.\n\n"
            "Do not include explanatory text outside the code fences.\n\n"
            "If CURRENT_PROJECT_CONTEXT is provided below, preserve existing file paths and "
            "behavior unless the user explicitly asks to change them."
        )

        # Refresh the workspace context so follow-up turns know what already exists.
        self._refresh_project_context()
        system = self.system_prompt
        if generation_rules not in system:
            system = system + "\n\n" + generation_rules

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

        cargo_result, pytest_rc, pytest_stdout, pytest_stderr, tests_passed = self._build_and_test_workspace()

        # If tests still fail, ask the LLM to repair the workspace using the failing output.
        max_heals = max(1, self.max_iterations)
        for heal_iter in range(max_heals):
            if tests_passed or cargo_result is None or cargo_result.returncode != 0:
                break
            self._progress(f"Healing test failures (attempt {heal_iter + 1}/{max_heals})...")
            try:
                bundle = bundle_workspace(self.output_dir, max_file_size_kb=50)
            except Exception as exc:
                logger.warning("Could not bundle workspace for healing: %s", exc)
                bundle = {"files": {}, "blueprint": None}
            context = format_context_block(bundle, fmt="xml") if bundle["files"] or bundle["blueprint"] else ""
            heal_system = (
                "You are repairing a generated Rust/Python workspace. "
                "The failing pytest output is shown below. "
                "Return the corrected files using the same Markdown code-fence format with file markers. "
                "Only include files that need to change; preserve the rest. "
                "Do not write explanatory text outside the code fences.\n\n"
                "Fix the implementation and/or tests so `pytest` passes. "
                "Ensure the Rust core exposes the same methods the tests call (e.g. `__len__`, `clear`, "
                "`mean`, `peak`, `std_dev`, `z_score`, `is_anomaly`, `process_batch`). "
                "For `process_batch`, always return `anomalies` as a list (empty list `[]` when none), never `None`."
            )
            if context:
                heal_system += "\n\n" + context
            error_text = f"{pytest_stderr}\n{pytest_stdout}"
            heal_messages = [
                {"role": "system", "content": heal_system},
                {
                    "role": "user",
                    "content": (
                        f"Original request: {text}\n\n"
                        f"Failing pytest output:\n```\n{error_text[:4000]}\n```\n\n"
                        "Return the corrected workspace files."
                    ),
                },
            ]
            try:
                heal_response = client.generate(heal_messages, temperature=0.2)
            except Exception as exc:
                self._progress(f"LLM healing call failed: {exc}")
                break
            if not heal_response:
                break
            heal_files = _parse_multifile_response(heal_response, self.output_dir)
            if not heal_files:
                break
            for path, content in heal_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                self._progress(f"Healed {path.relative_to(self.output_dir)}")
            cargo_result, pytest_rc, pytest_stdout, pytest_stderr, tests_passed = self._build_and_test_workspace()

        file_list = ExecutionReport(self.output_dir).filter_paths(
            sorted(str(p.relative_to(self.output_dir)) for p in self.output_dir.rglob("*") if p.is_file())
        )
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
            tier=Tier.FAST,
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

    def _heal_action(self) -> Dict[str, Any]:
        """Apply a deterministic self-healing patch to the failing source file."""
        if not self.last_terminal_log:
            return {"message": "No terminal error context. Run a command that fails, then ask me to 'fix error'."}

        from aero_forge.healing.evaluator import LogEvaluator

        evaluator = LogEvaluator()
        diagnosis = evaluator.evaluate_log(
            self.last_terminal_command or "",
            self.last_terminal_exit_code or 1,
            self.last_terminal_log,
        )
        self.last_evaluator_state = diagnosis

        if not diagnosis.get("healable", False):
            reason = diagnosis.get("reason") or "AST overlay is not safe for this error type."
            return {
                "message": f"This error can't be auto-patched: {reason} Ask me how to fix it instead.",
                "diagnosis": diagnosis,
                "status": "not_fixable",
            }

        target_file = diagnosis.get("target_file") or "main.py"
        target_path = self.output_dir / target_file
        if not target_path.is_file():
            return {
                "message": f"Could not locate target file '{target_file}' for healing.",
                "diagnosis": diagnosis,
            }

        original = target_path.read_text(encoding="utf-8")
        patched = try_auto_fix(self.last_terminal_log, original)
        if patched is None or patched == original:
            return {
                "message": "No deterministic patch matched this error.",
                "diagnosis": diagnosis,
                "status": "not_fixable",
            }

        target_path.write_text(patched, encoding="utf-8")
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=target_file,
                tofile=target_file,
            )
        )

        return {
            "message": f"Applied a patch to {target_file}. Re-run `{self.last_terminal_command}` to verify.",
            "status": "patched",
            "target_file": target_file,
            "diff": diff,
            "diagnosis": diagnosis,
            "rerun_command": self.last_terminal_command,
        }

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


class CopilotAgent(ChatSession):
    """Workspace-aware Copilot agent alias.

    This is the public entrypoint used by the web server.  It is a thin
    subclass of ``ChatSession`` that accepts ``workspace_path`` and exposes the
    same ``reply_structured`` / ``process`` interface.
    """


__all__ = [
    "ChatSession",
    "CopilotAgent",
    "SESSION_DIR",
    "get_session_metadata",
    "set_session_blueprint_metadata",
]
