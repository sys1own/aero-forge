"""User-facing error classification and helpers for the accelerate CLI."""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path
from typing import Optional

IO_ERROR = "Unsupported I/O operation detected. Aborting."


class AeroForgeError(Exception):
    """Base class for all Aero Forge domain errors."""


class ExportVerificationError(AeroForgeError):
    """Raised when a strict export fails pre-flight verification."""

    def __init__(
        self, message: str, verification: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.verification = verification or {}


class UnsupportedError(ValueError):
    """Raised when the source contains constructs we cannot compile."""

    def __init__(self, message: str, node: Optional[ast.AST] = None) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


def check_toolchain() -> None:
    """Verify that the Rust toolchain is available before building.

    If the toolchain is missing, attempt an isolated rustup bootstrap before
    failing.  The bootstrap respects ``AERO_FORGE_NO_RUST_BOOTSTRAP``.
    """
    from aero_forge.scaffold.cargo_runner import ensure_rust_toolchain

    env = ensure_rust_toolchain()
    if shutil.which("cargo", path=env.get("PATH")) and shutil.which(
        "rustc", path=env.get("PATH")
    ):
        os.environ.update(
            {
                k: v
                for k, v in env.items()
                if k in ("PATH", "CARGO_HOME", "RUSTUP_HOME") and k not in os.environ
            }
        )
        return
    path_dirs = env.get("PATH", "").split(os.pathsep)
    raise UserError(
        "Rust toolchain not found. Install Rust from https://rustup.rs/ or set "
        f"AERO_FORGE_NO_RUST_BOOTSTRAP=1 to disable auto-bootstrap. "
        f"Searched PATH directories: {', '.join(d for d in path_dirs if d)}"
    )


def locate_unsupported_node(source: str, message: str) -> Optional[ast.AST]:
    """Attempt to find the AST node that triggered a generic unsupported error."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    # Heuristic: search for the first occurrence of a statement/expression kind
    # mentioned in the error message (e.g. "Unsupported statement: With").
    prefix = "Unsupported "
    if not message.startswith(prefix):
        return None
    rest = message[len(prefix) :]
    if "statement" in rest:
        kind = rest.replace("statement:", "").replace("expression:", "").strip()
    elif "expression" in rest:
        kind = rest.replace("expression:", "").replace("statement:", "").strip()
    else:
        return None
    for node in ast.walk(tree):
        if type(node).__name__ == kind:
            return node
    return None


def classify_cargo_error(output: str) -> str:
    """Map raw cargo output to a concise, actionable message.

    The full cargo stderr is preserved in the exception/log; this function only
    adds a short classification so callers do not need to re-parse it.
    """
    out = output.lower()
    if (
        "e0428" in out
        or "defined multiple times" in out
        or "is defined multiple times" in out
    ):
        return "Name conflict in Rust code; please rename your function or file."
    if "m4" in out and ("not found" in out or "no usable" in out):
        return (
            "The C build tool m4 is missing. Install it with: sudo apt-get install m4"
        )
    if "linker" in out and "not found" in out:
        return "No linker found. Install a C toolchain (gcc/clang) on your system."
    if "could not choose a version" in out and "rustup" in out:
        return "Rust toolchain not configured. Run 'rustup default stable' or allow aero-forge to bootstrap Rust."
    if "error" in out:
        return "Rust compilation failed. The exact compiler output is included above."
    return "Cargo build failed. The exact compiler output is included above."


def format_unsupported_error(
    exc: UnsupportedError,
    source_path: Optional[Path] = None,
    source: Optional[str] = None,
) -> str:
    """Convert an UnsupportedError into a clear, location-aware message."""
    msg = exc.message
    if msg == "io":
        return IO_ERROR

    line = None
    node = exc.node
    if node is None and source is not None:
        node = locate_unsupported_node(source, msg)
    if node is not None:
        line = getattr(node, "lineno", None)

    parts = [f"Unsupported operation: {msg}."]
    if source_path:
        parts.append(f"File: {source_path}")
    if line:
        parts.append(f"Line: {line}")
    return " ".join(parts)


class UserError(Exception):
    """A runtime error that should be shown to the user without a traceback."""


class BuildStageError(UserError):
    """Raised when a native compilation or build stage fails with captured logs.

    Carrying the stage name and full logs lets callers route the failure to the
    deterministic error classifier / healer without being masked by downstream
    test collection failures.
    """

    def __init__(
        self,
        message: str,
        stage: str = "",
        logs: str = "",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.logs = logs


class SemanticRegressionError(Exception):
    """Reference and target executions diverged semantically."""

    def __init__(
        self,
        message: str,
        *,
        delta: int = 0,
        report: str = "",
    ) -> None:
        super().__init__(message)
        self.delta = delta
        self.report = report


class HeuristicWarning(UserError):
    """Raised when a function cannot be healed into a Tier-1/Tier-2 pathway.

    The caller must choose between manual refactoring or an explicit
    CPython fallback (e.g. ``precision_shield_mode='permissive'``).
    """
