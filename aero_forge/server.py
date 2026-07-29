"""Embedded HTTP server for Aero-Forge web integration."""

from __future__ import annotations

import asyncio
import fcntl
import functools
import difflib
import io
import json
import logging
import mimetypes
import os
import pty
import queue
import re
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time
import uuid
import zipfile
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp
import yaml
from aiohttp import web
from pydantic import ValidationError

from aero_forge import inspector as workspace_inspector
from aero_forge.blueprint import generate_blueprint_from_uploaded_repo
from aero_forge.bundle_repo import (
    ExportProfile,
    create_project_zip,
    zip_export_filename,
)
from aero_forge.chat import (
    ChatSession,
    get_session_metadata,
    set_session_blueprint_metadata,
)


def CopilotAgent(*args, **kwargs):
    """Workspace-aware Copilot agent entrypoint used by the web server.

    This is a thin dispatcher to the current ``ChatSession`` class so that
    server-side tests and callers can refer to ``CopilotAgent`` while still
    monkeypatching ``ChatSession`` when needed.  ``workspace_path`` and
    ``workspace_id`` are mapped to the positional ``output_dir`` argument.
    """
    workspace_path = kwargs.pop("workspace_path", kwargs.pop("workspace_id", None))
    if not args and workspace_path is not None:
        args = (workspace_path,)
    return ChatSession(*args, **kwargs)


from aero_forge.context_bundler import get_blueprint_status
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge import runner as sandbox_runner
from aero_forge.healing.context_builder import ContextBuilder
from aero_forge.orchestrator.orchestrator import purge_workspace_state
from aero_forge.healing.evaluator import LogEvaluator
from aero_forge.healing.llm_healer import LLMHealer, run_command
from aero_forge.healing.orchestrator import HealingOrchestrator
from aero_forge.healing.router import try_auto_fix
from aero_forge.healing.structural_merger import apply_overlay, MergeConflictError
from aero_forge.errors import UserError
from aero_forge.orchestrator.router import toolchains_for_intent
from aero_forge._native import run_aeroc
from aero_forge.builder.aeroc_compiler import (
    compile_blueprint_to_aeroc,
    compile_directory_to_aeroc,
)
from aero_forge.materializer import unpack_aeroc_file
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_CPP_RUST,
    INTENT_HYBRID_RUST_PYTHON,
    INTENT_PURE_PYTHON,
    INTENT_PURE_RUST,
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    StackClassification,
    classify_stack,
)
from aero_forge.accelerator.runtime import activate_runtime_native_acceleration
from aero_forge.ingestion.command_inspector import detect_runnable_commands
from aero_forge.ingestion.zip_parser import extract_zip_safely, generate_draft_v3_blueprint
from aero_forge.scaffold.module_guard import reify_missing_modules
from aero_forge.sandbox.manager import SandboxManager
from aero_forge.blueprint import (
    BlueprintV3,
    BlueprintV3Validator,
    LLMBlueprintSynthesizer,
    is_blueprint_ready,
    load_blueprint,
    write_v3_blueprint,
)
from aero_forge.blueprint.schema import ArtifactType, BuildArtifact, ContextState, GenerationMethod
from aero_forge.blueprint.validator import InvalidBlueprintError
from aero_forge.scaffold.pre_write_validator import BlueprintValidationError
from aero_forge.scaffold.aeroc_export import export_scaffold_zip
from aero_forge.scaffold.export_options import export_workspace
from aero_forge.scaffold.workspace import BlueprintRegenerator
from aero_forge.universal_builder import build_universal_project

logger = logging.getLogger("aero_forge.server")

DEFAULT_PORT = 8080


def _resolve_port(port: Optional[int] = None) -> int:
    """Return the effective port, honoring CLI args > PORT env > default."""
    if port is not None and port > 0:
        return port
    env_port = os.getenv("PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            logger.warning("Ignoring non-integer PORT environment variable: %r", env_port)
    return DEFAULT_PORT


def _resolve_llm_provider(body: Dict[str, Any]) -> str:
    """Return the effective LLM provider from the request body or environment."""
    return body.get("provider") or os.getenv("AERO_FORGE_LLM_PROVIDER") or "deepseek"

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, X-Api-Key, X-API-Key, Authorization",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}

_manager = SandboxManager()
_static_dir = Path(__file__).parent / "static"
_blueprint_templates_dir = Path(__file__).parent / "blueprint_templates"
_active_websockets: Dict[str, Any] = {}


def _classification_for_target(
    classification: StackClassification, target_language: str
) -> StackClassification:
    """Override the inferred stack classification when the user picks a target."""
    target = target_language.lower()
    if target in ("cpp", "hybrid_cpp_python"):
        architecture = INTENT_HYBRID_CPP_PYTHON
    elif target in ("rust", "hybrid_rust_python"):
        architecture = INTENT_HYBRID_RUST_PYTHON
    elif target in ("cpp_rust", "hybrid_cpp_rust"):
        architecture = INTENT_HYBRID_CPP_RUST
    elif target in ("python", "py"):
        architecture = INTENT_PURE_PYTHON
    elif target in ("tri_polyglot", "tri_polyglot_rust_cpp_python", "rust_cpp_python"):
        architecture = INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
    elif target in ("multi_crate_rust", "multi_crate"):
        architecture = INTENT_PURE_RUST
    elif target == "wasm":
        architecture = INTENT_PURE_RUST
    else:
        return classification
    features = classification.features
    if target == "wasm":
        features = sorted(set(features) | {"wasm"})
    return StackClassification(
        architecture=architecture,
        toolchains=toolchains_for_intent(architecture),
        languages=classification.languages,
        features=features,
    )
_active_ws_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _register_websocket(session_id: str, ws: Any) -> None:
    with _active_ws_lock:
        _active_websockets[session_id] = ws


def _unregister_websocket(session_id: str) -> None:
    with _active_ws_lock:
        _active_websockets.pop(session_id, None)


def _send_ws_heartbeat(session_id: str, phase: str, loop: asyncio.AbstractEventLoop) -> None:
    with _active_ws_lock:
        ws = _active_websockets.get(session_id)
    if ws is None or getattr(ws, "closed", False):
        return
    try:
        payload = json.dumps({"status": "building", "phase": phase})
        asyncio.run_coroutine_threadsafe(ws.send_str(payload), loop)
    except Exception as exc:
        logger.debug("Could not send build heartbeat: %s", exc)


def _set_event_loop() -> None:
    global _event_loop
    _event_loop = asyncio.get_running_loop()


async def _broadcast_tree(session_id: str) -> None:
    with _active_ws_lock:
        ws = _active_websockets.get(session_id)
    if ws is None or getattr(ws, "closed", False):
        return
    try:
        await ws.send_str(json.dumps({"type": "tree_updated"}))
    except Exception:
        pass


def _notify_tree_changed(session_id: str) -> None:
    if not session_id or _event_loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast_tree(session_id), _event_loop)
    except Exception:
        pass


def _session_dir(session_id: str) -> Path:
    return _manager.create_session_sandbox(session_id)


def _api_key_from_request(request: web.Request, body: Dict[str, Any]) -> Optional[str]:
    key = body.get("api_key") or body.get("apiKey")
    if key:
        return key
    key = request.headers.get("X-Api-Key") or request.headers.get("X-API-Key")
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]
    return None


def _phase_from_message(message: str) -> str:
    m = message.lower()
    if "compil" in m:
        return "cargo_compilation"
    if "test" in m:
        return "test_execution"
    if "generat" in m:
        return "code_generation"
    if m.startswith("build "):
        return "build_complete"
    return "in_progress"


async def _handle_build_async(request: web.Request) -> web.Response:
    """Run a build off the event loop and send progress heartbeats over the terminal WS."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = body.get("builder_prompt", body.get("prompt", "")).strip()
    if not prompt:
        return web.json_response({"error": "Missing 'prompt' or 'builder_prompt'"}, status=400)

    session_id = body.get("session_id") or str(uuid.uuid4())
    workspace_path = body.get("workspace_path") or body.get("output_dir")
    if workspace_path:
        output_dir = Path(workspace_path).expanduser().resolve()
    else:
        output_dir = _session_dir(session_id)
    session_dir = output_dir
    set_session_blueprint_metadata(session_id, is_building=True)
    variants = 3 if body.get("variants") else 1
    target_language = body.get("target_language", body.get("target", "auto"))
    acceleration_policy = body.get("acceleration_policy", "selective")
    architecture = body.get("architecture")
    config = ConfigOverride(
        llm_provider=body.get("provider"),
        api_key=_api_key_from_request(request, body),
        model=body.get("model"),
        max_retries=3,
    )

    loop = asyncio.get_running_loop()
    last_phase = {"phase": "code_generation"}

    def progress_callback(message: str) -> None:
        phase = _phase_from_message(message)
        last_phase["phase"] = phase
        _send_ws_heartbeat(session_id, phase, loop)

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
            _send_ws_heartbeat(session_id, last_phase["phase"], loop)

    heartbeat_task: asyncio.Task = asyncio.create_task(heartbeat())
    try:
        classification = _classification_for_target(
            classify_stack(prompt), target_language
        )
        if classification.architecture in (
            INTENT_HYBRID_RUST_PYTHON,
            INTENT_HYBRID_CPP_PYTHON,
            INTENT_HYBRID_CPP_RUST,
            INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
        ):
            universal_result = await asyncio.to_thread(
                build_universal_project,
                prompt,
                session_dir,
                project_name="generated",
                constraints=None,
                llm_provider=config.llm_provider or body.get("provider"),
                model=config.model or body.get("model"),
                max_retries=3,
                config_override=config,
                progress_callback=progress_callback,
                architecture=architecture or classification.architecture,
                acceleration_policy=acceleration_policy,
                workspace_path=output_dir,
            )
            result: Dict[str, Any] = {
                "build": universal_result,
                "files": universal_result.get("files", []),
            }
        else:
            result = await asyncio.to_thread(
                generate_and_build,
                prompt,
                output_dir=session_dir,
                project_name="generated",
                max_retries=3,
                max_iterations=5,
                variants=variants,
                build_kwargs={"max_workers": 1, "cache_enabled": False},
                config_override=config,
                progress_callback=progress_callback,
                target_language=target_language,
                acceleration_policy=acceleration_policy,
            )
        _notify_tree_changed(session_id)
        set_session_blueprint_metadata(session_id, source="auto_generated", auto_initialized=True)
        return web.json_response(
            _build_web_response(session_id, session_dir, result),
            headers=_CORS_HEADERS,
        )
    except Exception as exc:
        logger.exception("Build endpoint failed")
        return web.json_response(
            _build_web_response(
                session_id,
                session_dir,
                {"build": {"success": False, "error": str(exc)}},
            ),
            headers=_CORS_HEADERS,
        )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        set_session_blueprint_metadata(session_id, is_building=False)


def _send_json(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
    handler.end_headers()
    handler.wfile.write(body)


def _send_bytes(
    handler: BaseHTTPRequestHandler,
    status: int,
    data: bytes,
    content_type: str,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
    if headers:
        for key, value in headers.items():
            handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(data)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return b""
    return handler.rfile.read(length)


def _parse_json_body(handler: BaseHTTPRequestHandler) -> Any:
    raw = _read_body(handler)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _parse_multipart(body: bytes, boundary: bytes) -> Optional[bytes]:
    """Return the first file payload found in a multipart/form-data body."""
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers = part[:header_end].decode("utf-8", errors="ignore")
        if "filename=" in headers or "Content-Type:" in headers:
            return part[header_end + 4 :]
    return None


SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".aero",
    ".build_cache",
    ".overlays",
    ".variant_0",
    ".variant_1",
    ".variant_2",
    "target",
    ".cargo",
    ".pytest_cache",
}

_HUMAN_RELEVANT_EXTS = {
    ".py",
    ".rs",
    ".toml",
    ".aero",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".ini",
    ".cfg",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cc",
}

_BINARY_EXTS = {
    ".so",
    ".pyd",
    ".dylib",
    ".dll",
    ".wasm",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".whl",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
}

_SOURCE_EXTS = {
    ".py",
    ".rs",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".ts",
    ".js",
    ".go",
    ".java",
    ".kt",
    ".rb",
}


def _has_materialized_sources(workspace: Path) -> bool:
    """Return True when *workspace* contains actual source code files (not just blueprints)."""
    skip_names = {"blueprint.aero", "workspace_blueprint.yaml"}
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        rel = path.relative_to(workspace)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if path.suffix.lower() in _SOURCE_EXTS:
            return True
    return False


def _is_interesting_file(rel: Path) -> bool:
    """Return True for source files and final package outputs; skip build artifacts."""
    parts = rel.parts
    if any(part in SKIP_DIRS for part in parts):
        return False
    name = rel.name
    if name == ".gitkeep":
        return False
    if name == "Cargo.lock":
        return False
    if name.endswith(".aeroc") or name.endswith(".aerozip"):
        return False
    if parts[0] == "dist":
        return True
    ext = rel.suffix.lower()
    if ext in _HUMAN_RELEVANT_EXTS or ext in _BINARY_EXTS:
        return True
    return False


def _is_binary_file(path: Path) -> bool:
    """Return True for binary artifact files that should not be opened as text."""
    ext = path.suffix.lower()
    if ext in _BINARY_EXTS:
        return True
    try:
        sample = path.read_bytes()[:1024]
    except OSError:
        return False
    if b"\x00" in sample:
        return True
    if ext:
        return False
    return sample and not all(32 <= b < 127 or b in (9, 10, 13) for b in sample)


def _compile_workspace_aeroc(workspace: Path, output_path: Path) -> None:
    """Compile a workspace tree into a ``workspace.aeroc`` binary IR container."""
    blueprint_path = workspace / "blueprint.aero"
    if blueprint_path.is_file():
        try:
            data = yaml.safe_load(blueprint_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if str(data.get("metadata", {}).get("schema_version")) == "3.0.0":
            blueprint = BlueprintV3.load(blueprint_path)
            compile_blueprint_to_aeroc(blueprint, output_path, workspace=workspace)
            return
    compile_directory_to_aeroc(workspace, output_path)


def _run_workspace_build(session_id: str, workspace: Path, jobs: int = 4) -> Dict[str, Any]:
    """Compile the workspace to ``workspace.aeroc`` and execute it with the native daemon."""
    aeroc_path = workspace / "workspace.aeroc"
    _compile_workspace_aeroc(workspace, aeroc_path)
    run_aeroc(str(aeroc_path), str(workspace), jobs)
    return {"success": True, "aeroc": str(aeroc_path)}


def _build_tree(directory: Path, rel: Optional[Path] = None) -> Dict[str, Any]:
    """Return a nested JSON tree of files and directories, pruning build dirs."""
    rel = rel or Path(".")
    name = "." if rel == Path(".") else (rel.name or directory.name or ".")
    node: Dict[str, Any] = {
        "name": name,
        "type": "directory",
        "path": str(rel),
        "children": [],
    }
    for path in sorted(directory.iterdir()):
        child_rel = rel / path.name
        if path.is_dir():
            if path.name in SKIP_DIRS:
                continue
            node["children"].append(_build_tree(path, child_rel))
        else:
            if not _is_interesting_file(child_rel):
                continue
            node["children"].append(
                {
                    "name": path.name,
                    "type": "file",
                    "path": str(child_rel),
                    "size": path.stat().st_size,
                }
            )
    return node


def _build_file_list(directory: Path) -> List[Dict[str, Any]]:
    """Return a flat list of human-relevant file entries relative to *directory*."""
    files: List[Dict[str, Any]] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(directory)
        except ValueError:
            continue
        if not _is_interesting_file(rel):
            continue
        files.append(
            {
                "path": str(rel),
                "size": path.stat().st_size,
                "type": "file",
            }
        )
    return sorted(files, key=lambda x: x["path"])


def _build_web_response(
    session_id: str,
    session_dir: Path,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap a generation/build result with structured file updates, tree changes, and runnable commands."""
    files = _build_file_list(session_dir)
    tree = _build_tree(session_dir)
    commands = detect_runnable_commands(session_dir)
    build = result.get("build") or {}
    variants = result.get("variants", [])
    if variants:
        success_count = sum(
            1 for v in variants if (v.get("build") or {}).get("success")
        )
        if success_count == len(variants):
            status = "success"
        elif success_count > 0:
            status = "partial"
        else:
            status = "failure"
    else:
        status = "success" if build.get("success") else "failure"
    metadata = get_session_metadata(session_id)
    return {
        "session_id": session_id,
        "status": status,
        "files": files,
        "tree": tree,
        "commands": commands,
        "result": result,
        **metadata,
    }


def _resolve_file(session_dir: Path, file_path: str) -> Path:
    """Resolve ``file_path`` under ``session_dir`` and guard against traversal."""
    target = (session_dir / file_path).resolve()
    target.relative_to(session_dir.resolve())
    return target


def _canonicalize_chat_action(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert internal legacy/new action shapes into the canonical API action."""
    clean = result.get("clean_action")
    if clean and isinstance(clean, dict):
        return clean

    legacy = result.get("action")
    if not isinstance(legacy, dict):
        return None

    params = legacy.get("params") or {}
    prompt = (
        result.get("clean_prompt")
        or result.get("build_prompt")
        or result.get("suggested_build_prompt")
        or params.get("prompt")
        or params.get("build_prompt")
        or ""
    )
    if not prompt:
        return None

    parameters = result.get("parameters") or params.get("parameters") or {}
    target = parameters.get("target") or params.get("target") or "pure_python"
    acceleration = parameters.get("acceleration") or params.get("acceleration") or "Selective Acceleration (Auto-Detect Heavy Compute)"
    action_type = "build"
    if legacy.get("type") == "PROPOSE_BUILD" and params.get("blueprint"):
        action_type = "apply_blueprint"

    return {
        "type": action_type,
        "clean_prompt": prompt,
        "parameters": {"target": target, "acceleration": acceleration},
        "blueprint": params.get("blueprint"),
    }


class AeroForgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Aero-Forge web API."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def handle(self) -> None:
        """Handle a single HTTP request and close the connection."""
        try:
            self.handle_one_request()
        except Exception as exc:
            logger.exception("Request handler failed: %s", exc)
            try:
                self.send_error(500, message="Internal Server Error")
            except Exception:
                pass

    def finish(self) -> None:
        """Flush the in-memory response writer without closing shared buffers."""
        try:
            self.wfile.flush()
        except Exception:
            pass

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/api/files":
                return self._handle_files(query)
            if path == "/api/file-content":
                return self._handle_file_content(query)
            if path == "/api/files/download":
                return self._handle_files_download(query)
            if path == "/api/download-zip":
                return self._handle_download_zip(query)
            if path == "/api/blueprint-templates":
                return self._handle_blueprint_templates()
            if path == "/api/blueprint/status":
                return self._handle_blueprint_status(query)
            if path in ("/favicon.ico", "/static/logo.png"):
                return self._serve_static("/logo.png")

            return self._serve_static(path)
        except Exception as exc:
            logger.exception("GET %s failed", self.path)
            return _send_json(self, 500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/build" or path == "/api/builder/trigger":
                return self._handle_build()
            if path == "/api/aeroc/exec":
                return self._handle_aeroc_exec()
            if path == "/api/chat":
                return self._handle_chat()
            if path == "/api/blueprint/synthesize":
                return self._handle_blueprint_synthesize()
            if path == "/api/upload" or path == "/api/upload-zip":
                return self._handle_upload_zip()
            if path == "/api/unpack" or path == "/api/upload-aeroc":
                return self._handle_upload_aeroc()
            if path == "/api/files/upload":
                return self._handle_files_upload()
            if path == "/api/generate" or path == "/api/build":
                return self._handle_build()
            if path in ("/api/update", "/api/regenerate", "/api/workspace/regenerate_blueprint"):
                return self._handle_regenerate_blueprint()
            if path == "/api/save-file":
                return self._handle_save_file()
            if path == "/api/create-node":
                return self._handle_create_node()
            if path == "/api/files/create":
                return self._handle_create_node()
            if path == "/api/rename-node":
                return self._handle_rename_node()
            if path == "/api/delete-node":
                return self._handle_delete_node()
            if path == "/api/run":
                return self._handle_run()
            if path == "/api/workspace/accelerate":
                return self._handle_workspace_accelerate()
            if path == "/api/workspace/export":
                return self._handle_workspace_export()
            if path == "/api/workspace/download-aeroc":
                return self._handle_workspace_download_aeroc()
            if path == "/api/workspace/export-scaffold":
                return self._handle_workspace_export_scaffold()
            if path == "/api/workspace/evaluate-error":
                return self._handle_workspace_evaluate_error()
            if path == "/api/workspace/heal":
                return self._handle_workspace_heal()
            if path == "/api/workspace/heal/llm":
                return self._handle_workspace_heal_llm()
            if path == "/api/workspace/clean":
                return self._handle_workspace_clean()
            if path == "/api/workspace/regenerate_blueprint":
                return self._handle_regenerate_blueprint()
            if path == "/api/load-blueprint-template":
                return self._handle_load_blueprint_template()

            return _send_json(self, 404, {"error": "Not found"})
        except Exception as exc:
            logger.exception("POST %s failed", self.path)
            return _send_json(self, 500, {"error": str(exc)})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
        self.end_headers()

    def _api_key(self, body: Dict[str, Any]) -> Optional[str]:
        """Resolve API key from JSON body, X-Api-Key header, or Authorization header."""
        key = body.get("api_key")
        if key:
            return key
        key = self.headers.get("X-Api-Key") or self.headers.get("X-API-Key")
        if key:
            return key
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1]
        return None

    def _handle_build(self) -> None:
        try:
            body = _parse_json_body(self)
            prompt = body.get("builder_prompt", body.get("prompt", "")).strip()
            if not prompt:
                return _send_json(self, 400, {"error": "Missing 'prompt' or 'builder_prompt'"})

            session_id = body.get("session_id") or str(uuid.uuid4())
            workspace_path = body.get("workspace_path") or body.get("output_dir")
            if workspace_path:
                output_dir = Path(workspace_path).expanduser().resolve()
            else:
                output_dir = _session_dir(session_id)
            session_dir = output_dir
            set_session_blueprint_metadata(session_id, is_building=True)

            variants = 3 if body.get("variants") else 1
            target_language = body.get("target_language", body.get("target", "auto"))
            acceleration_policy = body.get("acceleration_policy", "selective")
            architecture = body.get("architecture")
            config = ConfigOverride(
                llm_provider=body.get("provider"),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )

            classification = _classification_for_target(
                classify_stack(prompt), target_language
            )
            if classification.architecture in (
                INTENT_HYBRID_RUST_PYTHON,
                INTENT_HYBRID_CPP_PYTHON,
                INTENT_HYBRID_CPP_RUST,
            ):
                universal_result = build_universal_project(
                    prompt,
                    session_dir,
                    project_name="generated",
                    constraints=None,
                    llm_provider=config.llm_provider,
                    model=config.model,
                    max_retries=3,
                    config_override=config,
                    architecture=architecture or classification.architecture,
                    acceleration_policy=acceleration_policy,
                    workspace_path=output_dir,
                )
                result: Dict[str, Any] = {
                    "build": universal_result,
                    "files": universal_result.get("files", []),
                }
            else:
                result = generate_and_build(
                    prompt,
                    output_dir=session_dir,
                    project_name="generated",
                    llm_provider=config.llm_provider,
                    model=config.model,
                    max_retries=3,
                    max_iterations=5,
                    variants=variants,
                    build_kwargs={"max_workers": 1, "cache_enabled": False},
                    config_override=config,
                    target_language=target_language,
                    acceleration_policy=acceleration_policy,
                )

            _notify_tree_changed(session_id)
            set_session_blueprint_metadata(session_id, source="auto_generated", auto_initialized=True)
            return _send_json(self, 200, _build_web_response(session_id, session_dir, result))
        except Exception as exc:  # pragma: no cover
            logger.exception("Build endpoint failed")
            return _send_json(
                self,
                200,
                _build_web_response(
                    session_id,
                    session_dir,
                    {"build": {"success": False, "error": str(exc)}},
                ),
            )
        finally:
            set_session_blueprint_metadata(session_id, is_building=False)

    def _handle_aeroc_exec(self) -> None:
        """Execute a workspace.aeroc container (or compile blueprint.aero first) with the native daemon."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)
            workspace_dir = body.get("workspace_dir", ".")
            file_path = body.get("path", "workspace.aeroc")
            jobs = int(body.get("jobs", 4))

            workspace = _resolve_file(session_dir, workspace_dir)
            target = _resolve_file(session_dir, file_path)

            if not target.is_file() and (file_path.endswith(".aero") or file_path.endswith(".py")):
                # Compile the workspace tree into workspace.aeroc first.
                aeroc_out = workspace / "workspace.aeroc"
                compile_directory_to_aeroc(workspace, aeroc_out)
                target = aeroc_out

            if not target.is_file():
                return _send_json(self, 404, {"error": f"aeroc file not found: {file_path}"})

            from aero_forge._native import run_aeroc

            run_aeroc(str(target), str(workspace), jobs)
            return _send_json(self, 200, {"status": "success", "executed": str(target), "workspace": str(workspace)})
        except Exception as exc:  # pragma: no cover
            logger.exception("aeroc exec failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_save_file(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            content = body.get("content")
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})
            if content is None:
                return _send_json(self, 400, {"error": "Missing 'content'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            target.parent.mkdir(parents=True, exist_ok=True)
            is_new_blueprint = target.name.lower() == "blueprint.aero" and not target.exists()
            target.write_text(content, encoding="utf-8")

            # Repair common LLM truncation in Rust/C/C++ files saved through the editor.
            if target.suffix.lower() in {".rs", ".cpp", ".c", ".h", ".hpp"}:
                from aero_forge.scaffold.syntax_guard import repair_file

                if repair_file(target):
                    logger.info("Repaired syntax truncation in saved file %s", target)

            if is_new_blueprint:
                set_session_blueprint_metadata(session_id, source="user_drop", auto_initialized=False)

            _notify_tree_changed(session_id)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "status": "saved",
                    "size": target.stat().st_size,
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Save-file endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _synthesize_if_raw(
        self,
        workspace: Path,
        config: ConfigOverride,
        session_id: Optional[str] = None,
    ) -> bool:
        """Run LLM blueprint synthesis when the workspace blueprint is still raw."""
        blueprint_path = workspace / "blueprint.aero"
        if not blueprint_path.is_file():
            return False
        try:
            bp = BlueprintV3.load(blueprint_path)
        except Exception as exc:
            logger.warning("Could not load blueprint for JIT synthesis: %s", exc)
            return False

        if bp.llm_context.state != "raw":
            return False

        if session_id:
            set_session_blueprint_metadata(session_id, is_synthesizing=True)
        try:
            provider = config.llm_provider or _resolve_llm_provider({})
            synthesizer = LLMBlueprintSynthesizer(
                provider=provider,
                model=config.model,
                api_key=config.api_key,
                config_override=config,
            )
            synthesizer.synthesize(workspace, draft=bp, output_path=blueprint_path)
            logger.info("JIT blueprint synthesis completed for %s", workspace)
            return True
        except Exception as exc:
            logger.warning("JIT blueprint synthesis failed for %s: %s", workspace, exc)
            return False
        finally:
            if session_id:
                set_session_blueprint_metadata(session_id, is_synthesizing=False)

    def _handle_chat(self) -> None:
        try:
            body = _parse_json_body(self)
            messages = body.get("messages", [])
            if not messages:
                # Backward compatibility: accept a single message field.
                text = body.get("message", "").strip()
                if not text:
                    return _send_json(self, 400, {"error": "Missing 'messages' or 'message'"})
                messages = [{"role": "user", "content": text}]

            session_id = body.get("session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)
            workspace_dir = (
                body.get("workspace_path")
                or body.get("workspace_dir")
                or body.get("workspace_id")
            )
            if workspace_dir:
                workspace_arg = Path(workspace_dir)
                if not workspace_arg.is_absolute():
                    workspace_arg = session_dir / workspace_arg
                # For safety, restrict chat workspace to the session sandbox.
                workspace_dir = workspace_arg.resolve()
            else:
                workspace_dir = session_dir

            config = ConfigOverride(
                llm_provider=body.get("provider"),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )

            # If the workspace blueprint has not been synthesized yet, run the
            # LLM synthesis pipeline before building the copilot context.
            self._synthesize_if_raw(workspace_dir, config, session_id=session_id)

            chat = CopilotAgent(
                workspace_path=workspace_dir,
                llm_provider=config.llm_provider,
                model=config.model,
                api_key=config.api_key,
                session_id=session_id,
                max_retries=3,
                config_override=config,
            )
            if messages:
                chat.messages = messages

            # Inject the latest terminal error context if the UI provides it.
            terminal_command = body.get("terminal_command")
            terminal_exit_code = body.get("terminal_exit_code")
            terminal_log_text = body.get("terminal_log_text")
            if (
                terminal_command is not None
                and terminal_exit_code is not None
                and terminal_log_text is not None
            ):
                chat.set_terminal_context(
                    terminal_command,
                    terminal_exit_code,
                    terminal_log_text,
                )

            # Use the last user message as the active turn text.
            user_text = [m for m in messages if m.get("role") == "user"][-1]["content"]

            # Chat is a Design & Advisory Engine. It never triggers builds or code
            # generation directly; it always returns a structured reply with an
            # optional SUGGEST_BUILD_PROMPT Action Card for the Builder tab. The
            # one exception is a deterministic AST self-heal request.
            lowered = user_text.lower()
            if any(phrase in lowered for phrase in ("fix error", "fix build", "apply self heal", "self heal", "heal")):
                command_action = chat.handle_command(user_text)
                if command_action is not None:
                    reply_text = chat._format_action_result(command_action, user_text)
                    result = {
                        "reply": reply_text,
                        "action": None,
                        "explanation": reply_text,
                        "has_suggestion": False,
                        "build_prompt": None,
                        "raw": reply_text,
                    }
                else:
                    result = chat.reply_structured(
                        user_text,
                        error_context=body.get("error_context"),
                    )
            else:
                result = chat.reply_structured(
                    user_text,
                    error_context=body.get("error_context"),
                )

            canonical_action = _canonicalize_chat_action(result)
            clean_prompt = (canonical_action or {}).get("clean_prompt") or result.get("build_prompt")
            response_payload = {
                "status": "success",
                "session_id": session_id,
                "type": result.get("type", "chat_message"),
                "display_text": result.get("display_text", result.get("reply", "")),
                "action": canonical_action,
                "clean_prompt": clean_prompt,
                "parameters": (canonical_action or {}).get("parameters", {}),
                "suggested_prompt": clean_prompt,
                # Backward-compatible legacy fields
                "reply": result.get("reply"),
                "message": result.get("message", result.get("reply", "")),
                "legacy_action": result.get("action"),
                "explanation": result.get("explanation", result.get("reply", "")),
                "has_suggestion": result.get("has_suggestion", bool(canonical_action)),
                "has_prompt": result.get("has_prompt", bool(clean_prompt)),
                "build_prompt": result.get("build_prompt") or clean_prompt,
                "suggested_build_prompt": result.get("suggested_build_prompt") or clean_prompt,
                "raw": result.get("raw", result.get("reply", "")),
                "messages": chat.messages,
            }
            return _send_json(self, 200, response_payload)
        except Exception as exc:  # pragma: no cover
            logger.exception("Chat endpoint failed")
            return _send_json(self, 500, {"status": "failed", "error": str(exc)})

    def _handle_blueprint_synthesize(self) -> None:
        """Synthesize a finalized Blueprint v3 from a session workspace or draft."""
        session_id: str = ""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)
            set_session_blueprint_metadata(session_id, is_synthesizing=True)
            workspace_dir = body.get("workspace_dir")
            if workspace_dir:
                workspace_arg = Path(workspace_dir)
                if not workspace_arg.is_absolute():
                    workspace_arg = session_dir / workspace_arg
                workspace = workspace_arg.resolve()
            else:
                workspace = session_dir

            draft_path = workspace / "blueprint.aero"
            draft: Optional[BlueprintV3] = None
            if draft_path.is_file():
                draft = BlueprintV3.load(draft_path)

            config = ConfigOverride(
                llm_provider=_resolve_llm_provider(body),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )
            synthesizer = LLMBlueprintSynthesizer(
                provider=config.llm_provider,
                model=config.model,
                api_key=config.api_key,
                config_override=config,
            )
            finalized = synthesizer.synthesize(
                workspace,
                draft=draft,
                output_path=draft_path,
            )
            BlueprintV3Validator(finalized.model_dump(mode="json"), workspace=workspace).check_exportable()

            return _send_json(
                self,
                200,
                {
                    "status": "finalized",
                    "session_id": session_id,
                    "path": str(draft_path),
                    "transferable": finalized.metadata.transferable,
                    "metadata": finalized.metadata.model_dump(mode="json"),
                },
            )
        except (
            yaml.YAMLError,
            ValueError,
            InvalidBlueprintError,
            BlueprintValidationError,
            ValidationError,
        ) as exc:
            logger.warning("Blueprint synthesize endpoint failed: %s. Attempting local fallback...", exc)
            try:
                workspace = locals().get("workspace")
                draft_path = locals().get("draft_path")
                if not workspace or not draft_path:
                    raise ValueError("Workspace or blueprint path not available for fallback")
                fallback = generate_draft_v3_blueprint(workspace)
                fallback.metadata.schema_version = "3.0.0"
                fallback.metadata.status = "finalized"
                fallback.metadata.generation_method = GenerationMethod.static_heuristic
                fallback.metadata.transferable = True
                fallback.metadata.llm_initialized = False
                fallback.llm_context.state = ContextState.synthesized
                if not fallback.build_pipeline:
                    fallback.build_pipeline.append(
                        BuildArtifact(
                            id="fallback_build",
                            type=ArtifactType.python_extension,
                            source_files=["main.py"],
                            output_path="dist/fallback_build",
                            description="Fallback artifact generated from workspace scan",
                        )
                    )
                write_v3_blueprint(fallback, draft_path)
                BlueprintV3Validator(fallback.model_dump(mode="json"), workspace=workspace).check_exportable()

                return _send_json(
                    self,
                    200,
                    {
                        "status": "finalized",
                        "session_id": session_id,
                        "path": str(draft_path),
                        "transferable": fallback.metadata.transferable,
                        "metadata": fallback.metadata.model_dump(mode="json"),
                    },
                )
            except Exception as fallback_exc:
                logger.exception("Local fallback blueprint generation failed")
                return _send_json(
                    self,
                    500,
                    {"status": "failed", "error": f"Synthesis failed and fallback failed: {fallback_exc}"},
                )
        except Exception as exc:
            logger.exception("Blueprint synthesize endpoint failed")
            return _send_json(self, 500, {"status": "failed", "error": str(exc)})
        finally:
            if session_id:
                set_session_blueprint_metadata(session_id, is_synthesizing=False)

    def _handle_blueprint_status(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        if not session_id:
            return _send_json(self, 400, {"error": "Missing 'session_id'"})

        session_dir = _session_dir(session_id)
        status = get_blueprint_status(session_dir)
        from aero_forge.blueprint import BlueprintV3, is_blueprint_ready

        try:
            blueprint_path = session_dir / "blueprint.aero"
            transferable = False
            ready = False
            if blueprint_path.is_file():
                bp = BlueprintV3.load(blueprint_path)
                transferable = bp.metadata.transferable
                ready = is_blueprint_ready(bp.model_dump())
        except Exception as exc:
            logger.exception("Failed to load blueprint for status")
            transferable = False
            ready = False

        return _send_json(
            self,
            200,
            {
                "session_id": session_id,
                "present": status["exists"],
                "status": status["status"] if status["exists"] else "missing",
                "transferable": transferable,
                "ready": ready,
                "schema_version": "3.0.0" if status["exists"] else None,
                "generation_method": status["generation_method"],
                "llm_initialized": status["llm_initialized"],
                "stale": status["stale"],
                "source_count": status["source_count"],
            },
        )

    def _handle_files(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        if not session_id:
            return _send_json(self, 400, {"error": "Missing 'session_id'"})

        session_dir = _session_dir(session_id)
        metadata = get_session_metadata(session_id)

        return _send_json(
            self,
            200,
            {
                "session_id": session_id,
                "tree": _build_tree(session_dir),
                **metadata,
            },
        )

    def _handle_file_content(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        file_path = _first(query, "path")
        if not session_id or not file_path:
            return _send_json(
                self, 400, {"error": "Missing 'session_id' and/or 'path'"}
            )

        session_dir = _session_dir(session_id)
        try:
            target = _resolve_file(session_dir, file_path)
        except ValueError:
            return _send_json(self, 400, {"error": "Invalid path"})

        if not target.is_file():
            return _send_json(self, 404, {"error": "File not found"})

        if _is_binary_file(target):
            content_type, _ = mimetypes.guess_type(str(target))
            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "binary": True,
                    "size": target.stat().st_size,
                    "mime_type": content_type or "application/octet-stream",
                },
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _send_json(self, 400, {"error": "File is not text-readable"})

        return _send_json(
            self,
            200,
            {
                "session_id": session_id,
                "path": file_path,
                "content": content,
            },
        )

    def _handle_upload_zip(self) -> None:
        try:
            body = _read_body(self)
            if not body:
                return _send_json(self, 400, {"error": "Empty body"})

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                match = re.search(r'boundary=([^;\s]+)', content_type)
                if not match:
                    return _send_json(self, 400, {"error": "Missing multipart boundary"})
                boundary = match.group(1).encode("utf-8")
                zip_bytes = _parse_multipart(body, boundary)
                if zip_bytes is None:
                    return _send_json(self, 400, {"error": "No file found in multipart body"})
            else:
                zip_bytes = body

            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            session_id = _first(query, "session_id") or str(uuid.uuid4())
            target_path = (_first(query, "target_path") or "").strip()
            session_dir = _session_dir(session_id)

            if target_path:
                if any(part == ".." for part in Path(target_path).parts):
                    return _send_json(self, 400, {"error": "Invalid target_path"})
                dest = session_dir / target_path
            else:
                dest = session_dir

            extract_zip_safely(zip_bytes, dest)

            # Repair truncated Rust/C/C++ sources before any build or bundling.
            from aero_forge.scaffold.syntax_guard import repair_workspace

            repair_workspace(session_dir)
            reify_missing_modules(session_dir)

            # Synthesize a v3.0 draft blueprint if none exists.
            blueprint_generated = False
            blueprint_path = session_dir / "blueprint.aero"
            has_sources = _has_materialized_sources(session_dir)
            if not blueprint_path.is_file():
                try:
                    from aero_forge.blueprint.schema import write_v3_blueprint
                    draft = generate_draft_v3_blueprint(session_dir)
                    write_v3_blueprint(draft, blueprint_path)
                    blueprint_generated = True
                except Exception as exc:
                    logger.warning("Could not auto-generate v3 blueprint for upload: %s", exc)

            # If the archive already contained source files, treat the workspace as
            # materialized so the frontend does not prompt to "initialize from blueprint".
            source_first = has_sources and not blueprint_generated
            auto_initialized = blueprint_generated or source_first
            blueprint_source = "zip_archive" if source_first else "auto_generated"

            # Refresh the chat session context with a compact workspace bundle so
            # subsequent chat turns can see the uploaded source tree.
            chat = ChatSession(session_dir, session_id=session_id)
            chat.blueprint_source = blueprint_source
            chat.auto_initialized = auto_initialized
            chat._save_session()

            _notify_tree_changed(session_id)

            commands = detect_runnable_commands(session_dir)

            message = (
                "Workspace loaded from ZIP archive"
                if source_first
                else "ZIP extracted & normalized to blueprint.aero"
            )
            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "success",
                    "files": _build_tree(session_dir),
                    "commands": commands,
                    "runnable_commands": commands,
                    "blueprint_source": blueprint_source,
                    "auto_initialized": auto_initialized,
                    "message": message,
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("Upload endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_upload_aeroc(self) -> None:
        """Accept a raw ``workspace.aeroc`` binary IR upload, unpack it, and build."""
        try:
            body = _read_body(self)
            if not body:
                return _send_json(self, 400, {"error": "Empty body"})

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                match = re.search(r'boundary=([^;\s]+)', content_type)
                if not match:
                    return _send_json(self, 400, {"error": "Missing multipart boundary"})
                boundary = match.group(1).encode("utf-8")
                aeroc_bytes = _parse_multipart(body, boundary)
                if aeroc_bytes is None:
                    return _send_json(self, 400, {"error": "No file found in multipart body"})
            else:
                aeroc_bytes = body

            if len(aeroc_bytes) < 8 or aeroc_bytes[:8] != b"AEROFOG\0":
                return _send_json(self, 400, {"error": "Invalid .aeroc file: missing AEROFOG magic"})

            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            session_id = _first(query, "session_id") or str(uuid.uuid4())
            target_path = (_first(query, "target_path") or "").strip()
            jobs = int(_first(query, "jobs") or "4")
            session_dir = _session_dir(session_id)

            if target_path:
                if any(part == ".." for part in Path(target_path).parts):
                    return _send_json(self, 400, {"error": "Invalid target_path"})
                target_dir = session_dir / target_path
            else:
                target_dir = session_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            aeroc_path = target_dir / "workspace.aeroc"
            aeroc_path.write_bytes(aeroc_bytes)

            try:
                unpack_aeroc_file(str(aeroc_path), str(target_dir))
            except Exception as exc:
                return _send_json(self, 400, {"error": f"Could not unpack .aeroc: {exc}"})
            finally:
                try:
                    aeroc_path.unlink(missing_ok=True)
                except OSError:
                    pass

            has_sources = _has_materialized_sources(target_dir)
            source_first = has_sources

            build_result: Dict[str, Any] = {"success": True, "skipped": source_first}
            if not source_first:
                try:
                    build_result = _run_workspace_build(session_id, target_dir, jobs=jobs)
                except Exception as exc:
                    build_result = {"success": False, "error": str(exc)}
                    logger.warning("Auto-build after .aeroc upload failed: %s", exc)

            # If the .aeroc already contained source code, treat it as a loaded
            # workspace and suppress the "initialize from blueprint" prompt.
            chat = ChatSession(target_dir, session_id=session_id)
            chat.blueprint_source = "aeroc_archive" if source_first else "user_drop"
            chat.auto_initialized = source_first
            chat._save_session()

            _notify_tree_changed(session_id)
            commands = detect_runnable_commands(target_dir)
            message = (
                "Workspace loaded from .aeroc archive"
                if source_first
                else "Aeroc extracted and build pipeline triggered"
            )
            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "success",
                    "build": build_result,
                    "files": _build_tree(session_dir),
                    "commands": commands,
                    "runnable_commands": commands,
                    "blueprint_source": chat.blueprint_source,
                    "auto_initialized": chat.auto_initialized,
                    "message": message,
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("Aeroc upload endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_create_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            is_dir = bool(body.get("is_dir", False))
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if target.exists():
                return _send_json(self, 409, {"error": "Node already exists"})

            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")

            _notify_tree_changed(session_id)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "is_dir": is_dir,
                    "status": "created",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Create-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_rename_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            old_path = body.get("old_path", "").strip()
            new_path = body.get("new_path", "").strip()
            if not session_id or not old_path or not new_path:
                return _send_json(self, 400, {"error": "Missing 'session_id', 'old_path', and/or 'new_path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                source = _resolve_file(session_dir, old_path)
                target = _resolve_file(session_dir, new_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if not source.exists():
                return _send_json(self, 404, {"error": "Source not found"})
            if target.exists():
                return _send_json(self, 409, {"error": "Target already exists"})

            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)

            _notify_tree_changed(session_id)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "old_path": old_path,
                    "new_path": new_path,
                    "status": "renamed",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Rename-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_delete_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if not target.exists():
                return _send_json(self, 404, {"error": "Node not found"})

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            _notify_tree_changed(session_id)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "status": "deleted",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Delete-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_files_download(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        file_path = _first(query, "path")
        if not session_id or not file_path:
            return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})

        session_dir = _session_dir(session_id)
        try:
            target = _resolve_file(session_dir, file_path)
        except ValueError:
            return _send_json(self, 400, {"error": "Invalid path"})

        if not target.is_file():
            return _send_json(self, 404, {"error": "File not found"})

        data = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        content_type = content_type or "application/octet-stream"
        return _send_bytes(
            self,
            200,
            data,
            content_type,
            {
                "Content-Disposition": f'attachment; filename="{target.name}"',
            },
        )

    def _handle_files_upload(self) -> None:
        """Upload one or more files (raw body or multipart) into the workspace."""
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            session_id = _first(query, "session_id") or ""
            target_path = (_first(query, "target_path") or "").strip() or "."
            filename = (_first(query, "filename") or "").strip()

            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _manager.create_session_sandbox(session_id)
            body = _read_body(self)
            if not body:
                return _send_json(self, 400, {"error": "Empty body"})

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type and not filename:
                match = re.search(r'boundary=([^;\s]+)', content_type)
                if match:
                    boundary = match.group(1).encode("utf-8")
                    file_bytes = _parse_multipart(body, boundary)
                    # Attempt to extract the filename from the first file part header.
                    parts = body.split(b"--" + boundary)
                    for part in parts:
                        if b"Content-Disposition:" in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                headers = part[:header_end].decode("utf-8", errors="ignore")
                                name_match = re.search(r'filename="([^"]+)"', headers)
                                if name_match:
                                    filename = name_match.group(1)
                                    body = file_bytes or b""
                                    break
                    else:
                        body = file_bytes or b""

            if not filename:
                return _send_json(self, 400, {"error": "Missing 'filename'"})

            if target_path and target_path != ".":
                if any(part == ".." for part in Path(target_path).parts):
                    return _send_json(self, 400, {"error": "Invalid target_path"})
                dest_dir = session_dir / target_path
            else:
                dest_dir = session_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            try:
                dest = dest.resolve()
                dest.relative_to(session_dir.resolve())
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid filename"})

            if filename.lower().endswith(".zip"):
                extract_zip_safely(body, dest_dir, archive_name=filename)
                # Auto-generate a v3 draft blueprint when a ZIP is dropped.
                blueprint_path = session_dir / "blueprint.aero"
                if not blueprint_path.is_file():
                    try:
                        from aero_forge.blueprint.schema import write_v3_blueprint
                        draft = generate_draft_v3_blueprint(session_dir)
                        write_v3_blueprint(draft, blueprint_path)
                    except Exception as exc:
                        logger.warning("Could not auto-generate v3 blueprint for file upload: %s", exc)
            else:
                is_new_blueprint = filename.lower() == "blueprint.aero" and not dest.exists()
                dest.write_bytes(body)
                if is_new_blueprint:
                    set_session_blueprint_metadata(session_id, source="user_drop", auto_initialized=False)

            _notify_tree_changed(session_id)
            commands = detect_runnable_commands(session_dir)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "success",
                    "path": str(dest.relative_to(session_dir)),
                    "tree": _build_tree(session_dir),
                    "commands": commands,
                    "runnable_commands": commands,
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Files upload endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_clean(self) -> None:
        """Reset a session sandbox to a clean workspace."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            purge_workspace_state(session_dir)
            _manager.clean_session_sandbox(session_id)
            session_dir = _manager.create_session_sandbox(session_id)

            _notify_tree_changed(session_id)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "cleaned",
                    "tree": _build_tree(session_dir),
                },
            )
        except Exception as exc:
            logger.exception("Workspace clean endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_regenerate_blueprint(self) -> None:
        """Wipe and rebuild a workspace from its ``blueprint.aero`` file."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            workspace_dir = body.get("workspace_dir") or str(session_dir)
            workspace_path = Path(workspace_dir).resolve()
            if not workspace_path.is_dir():
                return _send_json(
                    self, 400, {"error": f"Workspace not found: {workspace_dir}"}
                )

            blueprint_path = workspace_path / "blueprint.aero"
            if not blueprint_path.is_file():
                return _send_json(
                    self, 400, {"error": "blueprint.aero not found in workspace"}
                )

            blueprint = load_blueprint(blueprint_path)
            if not blueprint or not is_blueprint_ready(blueprint):
                return _send_json(
                    self,
                    400,
                    {
                        "status": "error",
                        "message": (
                            "Regeneration blocked: Blueprint is uninitialized. "
                            "Cannot overwrite existing workspace code."
                        ),
                    },
                )

            force_overwrite = bool(body.get("force_overwrite", False))
            if not force_overwrite:
                has_user_files = any(
                    e.name not in {"blueprint.aero", "workspace_blueprint.yaml", "workspace_blueprint.yml"}
                    and not e.name.startswith(".")
                    for e in workspace_path.iterdir()
                )
                if has_user_files:
                    return _send_json(
                        self,
                        400,
                        {
                            "status": "error",
                            "message": (
                                "Workspace is not empty. Use force_overwrite to regenerate."
                            ),
                        },
                    )

            # Purge caches, overlays, and healing state before regenerating so
            # stale state cannot corrupt the rebuilt workspace.
            purge_workspace_state(workspace_path)

            config = ConfigOverride(
                llm_provider=body.get("provider"),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )

            regenerator = BlueprintRegenerator(
                workspace_path,
                keep_backup=bool(body.get("keep_backup", False)),
                run_build=bool(body.get("run_build", False)),
                force_overwrite=bool(body.get("force_overwrite", False)),
                llm_provider=config.llm_provider,
                model=config.model,
                config_override=config,
            )
            result = regenerator.run()

            # The workspace has now been materialized from the blueprint.
            set_session_blueprint_metadata(
                session_id, source="user_drop", auto_initialized=True
            )

            _notify_tree_changed(session_id)
            commands = detect_runnable_commands(workspace_path)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": result.get("status", "partial"),
                    "errors": result.get("errors", []),
                    "logs": result.get("logs", []),
                    "backup_dir": result.get("backup_dir"),
                    "tree": _build_tree(workspace_path),
                    "commands": commands,
                    "runnable_commands": commands,
                    "blueprint_source": "user_drop",
                    "auto_initialized": True,
                },
            )
        except FileNotFoundError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except UserError as exc:
            return _send_json(
                self,
                400,
                {"status": "error", "message": str(exc)},
            )
        except Exception as exc:
            logger.exception("Regenerate blueprint endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_accelerate(self) -> None:
        """Activate runtime native acceleration or scaffold a PyO3 crate in the workspace.

        Streams step-by-step acceleration progress as NDJSON ``type: accel`` chunks
        and ends with a ``type: summary`` payload.  Toolchain events come directly
        from ``ToolchainManager`` and carry ``[TOOLCHAIN]`` / ``[ENV]`` prefixes.
        """
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            mode = body.get("mode", "runtime")
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})
            if mode not in {"runtime", "scaffold_pyo3"}:
                return _send_json(
                    self,
                    400,
                    {"error": "Invalid mode; expected 'runtime' or 'scaffold_pyo3'"},
                )

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )
        except Exception as exc:
            logger.exception("Workspace accelerate endpoint setup failed")
            return _send_json(self, 500, {"error": str(exc)})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def _write_chunk(obj):
            data = (json.dumps(obj) + "\n").encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _accel(level: str, prefix: str, message: str):
            _write_chunk({"type": "accel", "data": f"[{prefix}] {message}", "level": level})

        def _log_callback(level: str, prefix: str, message: str):
            _accel(level, prefix, message)

        from aero_forge.toolchain import ToolchainManager

        manager = ToolchainManager(session_dir, log_callback=_log_callback)

        try:
            _accel("info", "ACCEL", f"Initializing workspace acceleration (Mode: {mode})...")
            _accel("info", "ACCEL", "Scanning workspace root for build manifests...")
            commands = workspace_inspector.inspect_workspace(session_dir)
            cmd_labels = [c.get("cmd", c.get("label", "")) for c in commands]
            _accel("info", "ACCEL", f"Detected {len(commands)} runnable command(s): {cmd_labels}")

            native_active = False
            probe_command = "maturin develop" if mode == "scaffold_pyo3" else "runtime"
            try:
                manager.prepare_environment(probe_command)
            except RuntimeError as exc:
                _accel("error", "TOOLCHAIN", str(exc))
                raise

            if mode == "runtime":
                try:
                    from aero_forge import accelerator as accel_module

                    native_active = bool(accel_module.is_native())
                except Exception as exc:
                    _accel("warning", "ACCEL", f"Native acceleration engine not available: {exc}")
                if not native_active:
                    native_active = activate_runtime_native_acceleration(
                        session_dir, log_callback=_accel
                    )
                if native_active:
                    _accel("success", "ACCEL", "Native acceleration active (crates/native_core loaded).")
                else:
                    _accel("info", "ACCEL", "Native acceleration engine not active; using pure-Python fallback.")
            else:
                _accel("info", "ACCEL", "Scaffolding PyO3 / Cargo.toml bindings into workspace...")
                workspace_inspector.scaffold_pyo3_workspace(session_dir)
                _notify_tree_changed(session_id)
                _accel("success", "ACCEL", "PyO3 crate scaffolding complete.")
                commands = workspace_inspector.inspect_workspace(session_dir)
                cmd_labels = [c.get("cmd", c.get("label", "")) for c in commands]
                _accel("info", "ACCEL", f"Updated runnable command(s): {cmd_labels}")

            _write_chunk(
                {
                    "type": "summary",
                    "session_id": session_id,
                    "mode": mode,
                    "status": "accelerated",
                    "native_active": native_active,
                    "commands": commands,
                }
            )
            self.wfile.write(b"0\r\n\r\n")
        except Exception as exc:
            logger.exception("Workspace accelerate endpoint failed")
            try:
                _accel("error", "ACCEL", str(exc))
                _write_chunk({"type": "summary", "status": "error", "error": str(exc)})
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass

    def _handle_workspace_export(self) -> None:
        """Export the workspace as a ZIP according to the selected option flags."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            options = body.get("options", {})
            project_name = body.get("project_name", "aero-forge-export")
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            archive_bytes, filename = export_workspace(
                session_dir, options, project_name=project_name
            )
            return _send_bytes(
                self,
                200,
                archive_bytes,
                "application/zip",
                {"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            logger.exception("Workspace export endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_download_aeroc(self) -> None:
        """Serve the compiled binary IR container ``workspace.aeroc`` directly.

        If the container does not exist, compile it from ``blueprint.aero``,
        ``blueprint.py``, or the entire workspace tree. Build/materialization
        in progress or compilation failures block the export.
        """
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            project_name = body.get("project_name", "workspace").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            metadata = get_session_metadata(session_id)
            if metadata.get("is_building") or metadata.get("is_synthesizing"):
                return _send_json(
                    self,
                    409,
                    {"error": "Workspace build or LLM materialization is in progress; cannot export .aeroc now."},
                )

            aeroc_path = session_dir / "workspace.aeroc"
            try:
                _compile_workspace_aeroc(session_dir, aeroc_path)
            except Exception as exc:
                logger.exception("download-aeroc compilation failed")
                return _send_json(
                    self,
                    422,
                    {"error": f"Workspace failed compilation/validation: {exc}"},
                )

            if not aeroc_path.is_file():
                return _send_json(self, 404, {"error": "workspace.aeroc not found"})

            data = aeroc_path.read_bytes()
            return _send_bytes(
                self,
                200,
                data,
                "application/octet-stream",
                {"Content-Disposition": f'attachment; filename="{project_name}.aeroc"'},
            )
        except Exception as exc:
            logger.exception("download-aeroc endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_export_scaffold(self) -> None:
        """Export the workspace source tree as a Wavefront scaffold zip (``.aerozip``)."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            project_name = body.get("project_name", "aero-forge-export").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            archive_path = export_scaffold_zip(
                session_dir,
                output_path=session_dir / f"{project_name}.aerozip",
                project_name=project_name,
            )
            archive_bytes = archive_path.read_bytes()
            filename = archive_path.name
            return _send_bytes(
                self,
                200,
                archive_bytes,
                "application/zip",
                {"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            logger.exception("export-scaffold endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_evaluate_error(self) -> None:
        """Diagnose a terminal failure and decide whether it can be auto-healed."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            command = body.get("command", "")
            exit_code = body.get("exit_code", 1)
            log_text = body.get("log_text", "")
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            evaluator = LogEvaluator()
            diagnosis = evaluator.evaluate_log(command, exit_code, log_text)
            diagnosis["session_id"] = session_id
            return _send_json(self, 200, diagnosis)
        except Exception as exc:
            logger.exception("Workspace evaluate-error endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_workspace_heal(self) -> None:
        """Apply a smart heal (AST-first, then full-workspace LLM fallback)."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            command = body.get("command", "")
            exit_code = body.get("exit_code", 1)
            log_text = body.get("log_text", "")
            target_file = body.get("target_file", "").strip()
            force_llm = bool(body.get("force_llm", False))
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            provider = _resolve_llm_provider(body)
            model = body.get("model") or os.getenv("AERO_FORGE_MODEL")

            logs: list[str] = []

            def _log(level: str, prefix: str, message: str) -> None:
                line = f"[{level.upper()}] {prefix}: {message}"
                logs.append(line)
                logger.log(getattr(logging, level.upper(), logging.INFO), message)

            orchestrator = HealingOrchestrator(
                session_dir,
                llm_provider=provider,
                llm_model=model,
                log_callback=_log,
            )
            result = orchestrator.heal(
                error_logs=log_text,
                command=command,
                exit_code=exit_code,
                target_file=target_file or None,
                force_llm=force_llm,
            )
            if result.get("status") == "success":
                _notify_tree_changed(session_id)

            # Translate the orchestrator result into the frontend contract.
            response = {
                "session_id": session_id,
                "status": result.get("status", "failed"),
                "strategy_used": result.get("strategy_used"),
                "patched_files": result.get("patched_files", []),
                "error_message": result.get("error_message"),
                "target_file": result.get("target_file"),
                "diff": result.get("diff"),
                "diagnosis": result.get("diagnosis"),
                "attempts_exhausted": result.get("attempts_exhausted", False),
                "logs": logs,
            }
            return _send_json(self, 200, response)
        except Exception as exc:
            logger.exception("Workspace heal endpoint failed")
            return _send_json(
                self,
                200,
                {
                    "status": "failed",
                    "strategy_used": None,
                    "patched_files": [],
                    "error_message": str(exc),
                },
            )

    def _handle_workspace_heal_llm(self) -> None:
        """Apply an LLM-generated directive-based fix and re-run the command."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            command = body.get("command", "")
            exit_code = body.get("exit_code", 1)
            log_text = body.get("log_text", "")
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _session_dir(session_id)
            if not session_dir.is_dir():
                return _send_json(
                    self,
                    404,
                    {"error": f"Sandbox for session '{session_id}' does not exist"},
                )

            env = os.environ.copy()
            env["AERO_FORGE_SESSION"] = session_id
            env["AERO_FORGE_SESSION_DIR"] = str(session_dir)
            env["AERO_FORGE_ACCEL_LOG"] = str(session_dir / ".aero_forge_accel.log")

            evaluator = LogEvaluator()
            diagnosis = evaluator.evaluate_log(command, exit_code, log_text)

            logs: list[str] = []

            def _log(level: str, prefix: str, message: str) -> None:
                line = f"[{level.upper()}] {prefix}: {message}"
                logs.append(line)
                try:
                    with open(env["AERO_FORGE_ACCEL_LOG"], "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:
                    pass

            failure_context = {
                "command": command,
                "exit_code": exit_code,
                "log_text": log_text,
                "diagnosis": diagnosis,
            }

            _log("info", "HEAL_LLM", "Building workspace context...")
            ContextBuilder(session_dir).build_failure_context(command, exit_code, log_text, diagnosis)
            _log("info", "HEAL_LLM", "Workspace context packaged.")

            healer = LLMHealer(log_callback=_log)
            fix_result = healer.generate_and_apply_fix(session_dir, failure_context)
            if fix_result.get("status") != "success":
                return _send_json(
                    self,
                    200,
                    {
                        "session_id": session_id,
                        "status": "failed",
                        "verified": False,
                        "reason": fix_result.get("reason", "No directives generated."),
                        "diagnosis": diagnosis,
                        "logs": logs,
                    },
                )

            _log("info", "HEAL_LLM", f"Applied {len(fix_result['applied'])} directive(s).")

            resolved, proc_env, _ = asyncio.run(
                sandbox_runner.resolve_command(
                    command,
                    env=env,
                    sandbox_dir=session_dir,
                )
            )
            _log("info", "HEAL_LLM", f"Re-running: {resolved}")
            run_result = run_command(resolved, session_dir, env=proc_env, timeout=300)
            _log("info" if run_result["exit_code"] == 0 else "error", "HEAL_LLM", f"Re-run exit code: {run_result['exit_code']}")

            if run_result["exit_code"] == 0:
                _notify_tree_changed(session_id)
                return _send_json(
                    self,
                    200,
                    {
                        "session_id": session_id,
                        "status": "success",
                        "verified": True,
                        "output": run_result["output"],
                        "directives": fix_result["directives"],
                        "applied": fix_result["applied"],
                        "diagnosis": diagnosis,
                        "logs": logs,
                    },
                )

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "failed",
                    "verified": False,
                    "output": run_result["output"],
                    "reason": "Re-run still failed after applying LLM directives.",
                    "directives": fix_result["directives"],
                    "applied": fix_result["applied"],
                    "diagnosis": diagnosis,
                    "logs": logs,
                },
            )
        except Exception as exc:
            logger.exception("Workspace heal/llm endpoint failed")
            return _send_json(self, 200, {"status": "failed", "reason": str(exc), "verified": False})

    def _handle_blueprint_templates(self) -> None:
        """List available blueprint template names."""
        try:
            templates = sorted(
                p.name for p in _blueprint_templates_dir.iterdir() if p.is_file() and p.suffix == ".aero"
            )
            return _send_json(self, 200, {"templates": templates})
        except Exception as exc:
            logger.exception("Blueprint templates endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_load_blueprint_template(self) -> None:
        """Copy a blueprint template into the session workspace."""
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            name = body.get("name", "").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})
            if not name or "/" in name or "\\" in name or ".." in name:
                return _send_json(self, 400, {"error": "Invalid template name"})

            template = _blueprint_templates_dir / name
            if not template.is_file() or not str(template.resolve()).startswith(
                str(_blueprint_templates_dir.resolve())
            ):
                return _send_json(self, 404, {"error": "Template not found"})

            session_dir = _manager.create_session_sandbox(session_id)
            dest = session_dir / name
            dest.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
            _notify_tree_changed(session_id)
            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": str(dest.relative_to(session_dir)),
                    "tree": _build_tree(session_dir),
                },
            )
        except Exception as exc:
            logger.exception("Load blueprint template endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_run(self) -> None:
        """Run a Python entry file in the session sandbox and stream NDJSON output."""
        proc: Optional[subprocess.Popen] = None
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            if not session_id:
                return _send_json(self, 400, {"error": "Missing 'session_id'"})

            session_dir = _manager.create_session_sandbox(session_id)

            file_path = body.get("path", "").strip()
            if not file_path:
                for candidate in ("src/main.py", "main.py"):
                    if (session_dir / candidate).is_file():
                        file_path = candidate
                        break
            if not file_path:
                for py_file in sorted(session_dir.rglob("*.py")):
                    rel = str(py_file.relative_to(session_dir))
                    if "test_" in rel or "/tests/" in rel or "_test" in rel:
                        continue
                    file_path = rel
                    break
            if not file_path:
                return _send_json(self, 400, {"error": "No Python entry file found"})

            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if not target.is_file():
                return _send_json(self, 404, {"error": "Entry file not found"})

            from aero_forge.toolchain import ToolchainManager

            manager = ToolchainManager(session_dir)
            manager.env = os.environ.copy()
            manager.env["AERO_FORGE_SESSION"] = session_id
            manager.env["AERO_FORGE_SESSION_DIR"] = str(session_dir)
            manager.env["AERO_FORGE_ACCEL_LOG"] = str(session_dir / ".aero_forge_accel.log")
            manager.env["PYTHONUNBUFFERED"] = "1"
            manager.prepare_environment("python")
            env = manager.env
            python_exe = manager._venv_python()

            start = time.time()
            max_duration = float(os.environ.get("AERO_FORGE_RUN_TIMEOUT", "120"))

            proc = subprocess.Popen(
                [python_exe, "-u", str(target)],
                cwd=str(session_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )

            # Set stdout and stderr to non-blocking so reader threads never hang.
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    fcntl.fcntl(pipe, fcntl.F_SETFL, fcntl.fcntl(pipe, fcntl.F_GETFL) | os.O_NONBLOCK)

            q: queue.Queue = queue.Queue()
            buffers: Dict[str, bytes] = {"stdout": b"", "stderr": b""}

            def reader(pipe, tag):
                fd = pipe.fileno()
                while True:
                    try:
                        data = os.read(fd, 4096)
                    except (BlockingIOError, OSError):
                        time.sleep(0.05)
                        continue
                    if not data:
                        break
                    buffers[tag] += data
                    while b"\n" in buffers[tag]:
                        line, _, buffers[tag] = buffers[tag].partition(b"\n")
                        q.put((tag, line.decode("utf-8", errors="replace")))
                if buffers[tag]:
                    q.put((tag, buffers[tag].decode("utf-8", errors="replace")))
                pipe.close()

            threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True).start()

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def _write_chunk(obj):
                data = (json.dumps(obj) + "\n").encode("utf-8")
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            accel_log_path = Path(env["AERO_FORGE_ACCEL_LOG"])
            accel_offset: int = 0

            def _emit_accel_lines() -> None:
                nonlocal accel_offset
                if not accel_log_path.is_file():
                    return
                with accel_log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(accel_offset)
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        level = "info"
                        if line.startswith("[") and "]" in line:
                            level = line[1:line.index("]")].lower() or "info"
                        _write_chunk({"type": "accel", "data": line, "level": level})
                    accel_offset = f.tell()

            finished = False
            while True:
                try:
                    tag, line = q.get(timeout=0.1)
                    _write_chunk({"type": tag, "data": line})
                except queue.Empty:
                    if proc.poll() is not None:
                        _emit_accel_lines()
                        if finished:
                            break
                        finished = True
                    elif time.time() - start > max_duration:
                        logger.warning("Run for %s timed out after %ss", file_path, max_duration)
                        proc.kill()
                        _emit_accel_lines()

            duration = (time.time() - start) * 1000
            _write_chunk(
                {
                    "type": "summary",
                    "exit_code": proc.returncode,
                    "duration_ms": round(duration, 1),
                    "file": file_path,
                }
            )
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception as exc:
            logger.exception("Run endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass

    def _handle_download_zip(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        if not session_id:
            return _send_json(self, 400, {"error": "Missing 'session_id'"})

        session_dir = _session_dir(session_id)
        include_native = _first(query, "include_native_crate") or ""
        profile = (
            ExportProfile.ACCELERATED_PYO3
            if include_native.lower() in ("1", "true", "yes", "on")
            else ExportProfile.STANDARD
        )

        if not session_dir.is_dir():
            return _send_json(self, 404, {"error": f"Sandbox for session '{session_id}' does not exist"})

        archive_bytes = create_project_zip(session_dir, profile=profile)
        filename = zip_export_filename(profile)

        return _send_bytes(
            self,
            200,
            archive_bytes,
            "application/zip",
            {
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        local_path = _static_dir / path.lstrip("/")
        if not local_path.is_file() or not str(local_path.resolve()).startswith(
            str(_static_dir.resolve())
        ):
            return _send_json(self, 404, {"error": "Not found"})

        content_type, _ = mimetypes.guess_type(str(local_path))
        content_type = content_type or "application/octet-stream"
        data = local_path.read_bytes()
        return _send_bytes(self, 200, data, content_type)


def _first(query: Dict[str, List[str]], key: str) -> Optional[str]:
    values = query.get(key)
    return values[0] if values else None


def _set_pty_size(master_fd: int, cols: int, rows: int) -> None:
    """Set the PTY window size so tools like top, htop and editors render."""
    try:
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError as exc:
        logger.debug("Could not set PTY size: %s", exc)


def _child_setup(master_fd: int, slave_fd: int) -> None:
    """Close the PTY master fd in the child before exec'ing the shell."""
    try:
        os.close(master_fd)
    except OSError:
        pass


async def _handle_terminal(websocket: Any) -> None:
    """Spawn a per-session subshell over a PTY and relay I/O to the WebSocket."""
    request_path = websocket.request.path if websocket.request else "/ws/terminal"
    parsed = urlparse(request_path)
    if parsed.path != "/ws/terminal":
        await websocket.close(code=1002, reason="Invalid path")
        return

    query = parse_qs(parsed.query)
    session_id = _first(query, "session_id") or str(uuid.uuid4())
    session_dir = _manager.create_session_sandbox(session_id)

    shell = shutil.which("bash") or shutil.which("sh")
    if not shell:
        await websocket.close(code=1011, reason="No shell available on this system")
        return

    master_fd: int = -1
    process: Optional[Any] = None
    reader_added = False
    loop = asyncio.get_running_loop()

    try:
        master_fd, slave_fd = pty.openpty()
        os.set_blocking(master_fd, False)

        from aero_forge.toolchain import ToolchainManager

        manager = ToolchainManager(session_dir)
        manager.env["TERM"] = "xterm-256color"
        manager.env["AERO_FORGE_SESSION"] = session_id
        manager.env["AERO_FORGE_SESSION_DIR"] = str(session_dir)

        # Ensure the shell's `python` resolves to the same interpreter that built
        # native extensions. If pyenv is active, force the system version (the
        # one running this server) so compiled C-ABI artifacts load correctly.
        manager.env["PYENV_VERSION"] = "system"

        await loop.run_in_executor(None, manager.prepare_environment, "python")
        env = manager.env

        process = await asyncio.create_subprocess_exec(
            shell,
            "-i",
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=session_dir,
            env=env,
            start_new_session=True,
            preexec_fn=lambda: _child_setup(master_fd, slave_fd),
        )
        try:
            os.close(slave_fd)
        except OSError:
            pass

        _set_pty_size(master_fd, 80, 24)

        async def _send_to_client(data: bytes) -> None:
            try:
                await websocket.send(data)
            except Exception:
                pass

        def _on_master_readable() -> None:
            try:
                data = os.read(master_fd, 4096)
            except (BlockingIOError, OSError):
                return
            if data:
                asyncio.create_task(_send_to_client(data))
            else:
                loop.remove_reader(master_fd)

        loop.add_reader(master_fd, _on_master_readable)
        reader_added = True

        async def _wait_for_process() -> None:
            if process is None:
                return
            try:
                await process.wait()
            except Exception:
                pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        asyncio.create_task(_wait_for_process())

        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            if not data:
                continue

            # Resize messages are sent by xterm.js as JSON {cols, rows}
            if data.startswith(b"{"):
                try:
                    payload = json.loads(data.decode("utf-8"))
                    if isinstance(payload, dict) and "cols" in payload and "rows" in payload:
                        _set_pty_size(master_fd, int(payload["cols"]), int(payload["rows"]))
                        continue
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            try:
                os.write(master_fd, data)
            except (BlockingIOError, OSError):
                break

    except Exception as exc:
        logger.exception("Terminal handler error: %s", exc)
    finally:
        if reader_added and master_fd >= 0:
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2)
            except Exception:
                pass
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass


class _FakeSocket:
    """Stand-in for a TCP socket when driving BaseHTTPRequestHandler in memory."""

    def __init__(self, request_bytes: bytes) -> None:
        self.r = io.BytesIO(request_bytes)
        self.w = io.BytesIO()

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> Any:
        if "r" in mode:
            return self.r
        if "w" in mode:
            return self.w
        raise ValueError(mode)

    def settimeout(self, *args: Any) -> None:
        pass

    def setsockopt(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sendall(self, data: bytes) -> int:
        return self.w.write(data)

    def close(self) -> None:
        pass

    def shutdown(self, *args: Any) -> None:
        pass


class _FakeServer:
    """Minimal server object required by BaseHTTPRequestHandler."""

    def __init__(self, server_address: Any) -> None:
        self.server_address = server_address


class _HttpResponseSock:
    """Wrapper that lets http.client.HTTPResponse read from a BytesIO buffer."""

    def __init__(self, fp: io.BytesIO) -> None:
        self.fp = fp

    def makefile(self, *args: Any, **kwargs: Any) -> io.BytesIO:
        return self.fp

    def close(self) -> None:
        pass


def _parse_http_response(response_bytes: bytes) -> web.Response:
    """Parse a raw HTTP response produced by BaseHTTPRequestHandler into an aiohttp response."""
    sock = _HttpResponseSock(io.BytesIO(response_bytes))
    response = HTTPResponse(sock)
    response.begin()
    body = response.read()
    headers: Dict[str, str] = {}
    for name, value in response.getheaders():
        if name.lower() in {"transfer-encoding", "content-length", "connection", "date", "server"}:
            continue
        headers[name] = value
    return web.Response(status=response.status, reason=response.reason, headers=headers, body=body)


async def _build_raw_request(request: web.Request) -> bytes:
    """Convert an aiohttp request into raw HTTP bytes for BaseHTTPRequestHandler."""
    body = await request.read()
    raw = f"{request.method} {request.raw_path} HTTP/1.1\r\n".encode()
    raw += f"Host: {request.host}\r\n".encode()
    for name, value in request.headers.items():
        if name.lower() in {"host", "connection"}:
            continue
        raw += f"{name}: {value}\r\n".encode()
    raw += b"Connection: close\r\n"
    if body:
        raw += f"Content-Length: {len(body)}\r\n".encode()
    raw += b"\r\n"
    raw += body
    return raw


def _run_http_handler(raw_request: bytes, port: int) -> bytes:
    """Execute the existing BaseHTTPRequestHandler against an in-memory socket."""
    sock = _FakeSocket(raw_request)
    server = _FakeServer(("", port))
    handler = AeroForgeHandler(sock, ("127.0.0.1", 0), server)
    try:
        handler.handle()
    except Exception as exc:
        logger.exception("HTTP handler failed: %s", exc)
        sock.w.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
    return sock.w.getvalue()


async def _aiohttp_http_handler(request: web.Request, port: int) -> web.Response:
    """Route any non-WebSocket HTTP request through the existing handler stack."""
    _set_event_loop()
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_CORS_HEADERS)
    if request.path == "/api/build" or request.path == "/api/builder/trigger":
        return await _handle_build_async(request)
    raw = await _build_raw_request(request)
    loop = asyncio.get_event_loop()
    response_bytes = await loop.run_in_executor(None, _run_http_handler, raw, port)
    return _parse_http_response(response_bytes)


class _AioWSAdapter:
    """Make an aiohttp WebSocketResponse look like the websocket object _handle_terminal expects."""

    def __init__(self, request: web.Request, ws: web.WebSocketResponse) -> None:
        self.request = type("Request", (), {"path": str(request.rel_url)})()
        self._ws = ws

    async def send(self, data: Any) -> None:
        if isinstance(data, bytes):
            await self._ws.send_bytes(data)
        elif isinstance(data, str):
            await self._ws.send_str(data)
        else:
            await self._ws.send_str(str(data))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, message=reason.encode("utf-8") if reason else b"")

    def __aiter__(self) -> Any:
        return self._aiter()

    async def _aiter(self) -> Any:
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT or msg.type == aiohttp.WSMsgType.BINARY:
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                break


async def _aiohttp_ws_handler(request: web.Request) -> web.WebSocketResponse:
    """Accept WebSocket upgrades on the same HTTP port and attach the terminal PTY."""
    _set_event_loop()
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    query = parse_qs(request.query_string)
    session_id = _first(query, "session_id") or str(uuid.uuid4())
    _register_websocket(session_id, ws)
    adapter = _AioWSAdapter(request, ws)
    try:
        await _handle_terminal(adapter)
    except Exception as exc:
        logger.exception("Terminal handler failed: %s", exc)
    finally:
        _unregister_websocket(session_id)
    return ws


async def _handle_terminal_run_async(request: web.Request) -> web.StreamResponse:
    """Run an arbitrary shell command inside a session sandbox and stream NDJSON output."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400, headers=_CORS_HEADERS)

    session_id = (body.get("session_id") or "").strip()
    command = (body.get("command") or "").strip()
    if not session_id:
        return web.json_response({"error": "Missing 'session_id'"}, status=400, headers=_CORS_HEADERS)
    if not command:
        return web.json_response({"error": "Missing 'command'"}, status=400, headers=_CORS_HEADERS)

    session_dir = _manager.create_session_sandbox(session_id)
    timeout = float(os.environ.get("AERO_FORGE_TERMINAL_TIMEOUT", "60"))

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "application/x-ndjson",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    env = os.environ.copy()
    try:
        command, env, logs = await sandbox_runner.resolve_command(command, env, session_dir)
    except Exception as exc:
        await response.write((json.dumps({"type": "stderr", "data": f"Toolchain resolution failed: {exc}"}) + "\n").encode("utf-8"))
        await response.write((json.dumps({"type": "summary", "exit_code": -1, "duration_ms": 0, "cwd": str(session_dir)}) + "\n").encode("utf-8"))
        await response.write_eof()
        return response

    for level, prefix, message in logs:
        await response.write((json.dumps({"type": "accel", "data": f"[{prefix}] {message}", "level": level}) + "\n").encode("utf-8"))

    start = time.time()

    def _wave_log(level: str, prefix: str, message: str) -> None:
        payload = json.dumps({"type": "accel", "data": f"[{prefix}] {message}", "level": level}) + "\n"
        # Schedule the write on the response writer; safe because we are inside the handler coroutine.
        data = payload.encode("utf-8")
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda d=data: asyncio.create_task(response.write(d)))

    try:
        results = await sandbox_runner.run_wavefront_tasks_async(
            [command],
            sandbox_dir=session_dir,
            env=env,
            log_callback=_wave_log,
        )
        result = results[0]
    except Exception as exc:
        await response.write((json.dumps({"type": "stderr", "data": f"Wavefront execution failed: {exc}"}) + "\n").encode("utf-8"))
        await response.write((json.dumps({"type": "summary", "exit_code": -1, "duration_ms": 0, "cwd": str(session_dir)}) + "\n").encode("utf-8"))
        await response.write_eof()
        return response

    for line in result.get("stdout", "").splitlines():
        await response.write((json.dumps({"type": "stdout", "data": line}) + "\n").encode("utf-8"))
    for line in result.get("stderr", "").splitlines():
        await response.write((json.dumps({"type": "stderr", "data": line}) + "\n").encode("utf-8"))
    if result.get("timed_out"):
        await response.write((json.dumps({"type": "stderr", "data": f"Command timed out after {timeout}s"}) + "\n").encode("utf-8"))

    duration = (time.time() - start) * 1000
    summary = json.dumps({
        "type": "summary",
        "exit_code": result["returncode"],
        "duration_ms": round(duration, 1),
        "cwd": str(session_dir),
    }) + "\n"
    await response.write(summary.encode("utf-8"))
    await response.write_eof()
    return response


async def _handle_blueprint_async(request: web.Request) -> web.Response:
    """Return the parsed ``blueprint.aero`` for the active session, exposing verification nodes."""
    query = parse_qs(request.query_string)
    session_id = _first(query, "session_id") or ""
    if not session_id:
        return web.json_response({"error": "Missing 'session_id'"}, status=400, headers=_CORS_HEADERS)

    session_dir = _manager.create_session_sandbox(session_id)
    blueprint_path = session_dir / "blueprint.aero"
    if not blueprint_path.is_file():
        return web.json_response({"error": "Blueprint not found"}, status=404, headers=_CORS_HEADERS)

    try:
        data = yaml.safe_load(blueprint_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return web.json_response({"error": f"Failed to parse blueprint: {exc}"}, status=500, headers=_CORS_HEADERS)

    return web.json_response(
        {
            "session_id": session_id,
            "metadata": data.get("metadata", {}),
            "execution_strategy": data.get("execution_strategy", {}),
            "abi_contracts": data.get("abi_contracts", []),
            "module_graph": data.get("module_graph", []),
            "verification_nodes": data.get("verification_nodes", []),
            "manifest": data.get("manifest", []),
        },
        headers=_CORS_HEADERS,
    )


class AioForgeServer:
    """Combined HTTP + WebSocket server using aiohttp on a single port."""

    DEFAULT_HOST = "0.0.0.0"
    MAX_PORT_RETRIES = 10

    def __init__(self, port: Optional[int] = None, host: str = DEFAULT_HOST) -> None:
        self.requested_port = _resolve_port(port)
        self.port = self.requested_port
        self.host = host
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._serve_error: Optional[Exception] = None
        self.runner: Optional[web.AppRunner] = None
        self.app = web.Application()
        self.app.router.add_get("/ws/terminal", _aiohttp_ws_handler)
        self.app.router.add_post("/api/terminal/run", _handle_terminal_run_async)
        self.app.router.add_get("/api/blueprint", _handle_blueprint_async)
        self.app.router.add_route("*", "/{tail:.*}", functools.partial(_aiohttp_http_handler, port=self.port))

    async def _serve(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site: Optional[web.TCPSite] = None
        try_port = self.requested_port
        for attempt in range(self.MAX_PORT_RETRIES):
            try:
                site = web.TCPSite(self.runner, host=self.host, port=try_port)
                await site.start()
                break
            except OSError as exc:
                if exc.errno in (98, 48) and attempt < self.MAX_PORT_RETRIES - 1:
                    logger.warning(
                        "Port %s (%s) is already in use; trying port %s",
                        try_port,
                        self.host,
                        try_port + 1,
                    )
                    try_port += 1
                    continue
                logger.error(
                    "Could not bind server to %s:%s after %d attempt(s): %s",
                    self.host,
                    try_port,
                    attempt + 1,
                    exc,
                )
                raise
        if site is None:
            raise RuntimeError("Failed to bind server after exhausting port retries")
        _set_event_loop()
        # Capture the actual bound port in case the OS assigned a different one.
        if site._server and site._server.sockets:
            self.port = site._server.sockets[0].getsockname()[1]
        else:
            self.port = try_port
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()
        await self.runner.cleanup()

    def serve_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except OSError as exc:
            self._serve_error = exc
            logger.error("Server failed to start: %s", exc)
        finally:
            self._loop.close()

    def shutdown(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def server_close(self) -> None:
        pass

    @property
    def server_address(self) -> tuple:
        return (self.host, self.port)


def make_server(port: Optional[int] = None, host: str = AioForgeServer.DEFAULT_HOST) -> AioForgeServer:
    """Return an aiohttp-based server bound to the given port."""
    return AioForgeServer(port=port, host=host)


def run_server(
    port: Optional[int] = None,
    host: str = AioForgeServer.DEFAULT_HOST,
    open_browser: bool = True,
) -> None:
    """Start the web server and optionally open the user's browser."""
    _static_dir.mkdir(parents=True, exist_ok=True)
    server = make_server(port, host)

    def serve() -> None:
        try:
            server.serve_forever()
        finally:
            server.server_close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    # Wait briefly for the server to bind so the URL reflects the actual port.
    import time
    for _ in range(50):
        if server.port != server.requested_port or server._serve_error is not None:
            break
        time.sleep(0.05)

    if server._serve_error is not None:
        logger.error("Aero-Forge web server failed to start on %s", server.server_address)
        return

    url = f"http://localhost:{server.port}"
    logger.info("Aero-Forge web server running at %s (HTTP + WebSocket on one port)", url)

    if open_browser:
        import webbrowser

        webbrowser.open(url)

    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("Shutting down web server...")
    finally:
        server.shutdown()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Aero-Forge web dashboard server")
    parser.add_argument("--host", default=AioForgeServer.DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_server(port=args.port, host=args.host, open_browser=not args.no_browser)
