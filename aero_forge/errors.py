"""User-facing error classification and helpers for the accelerate CLI."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Optional

IO_ERROR = "Unsupported I/O operation detected. Aborting."


class UnsupportedError(ValueError):
    """Raised when the source contains constructs we cannot compile."""

    def __init__(self, message: str, node: Optional[ast.AST] = None) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


def check_toolchain() -> None:
    """Verify that cargo and rustc are installed."""
    missing = [tool for tool in ("cargo", "rustc") if not shutil.which(tool)]
    if missing:
        raise UserError(
            f"Missing Rust toolchain: {', '.join(missing)}. "
            "Install Rust from https://rustup.rs/ and ensure cargo/rustc are on your PATH."
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
    """Map raw cargo output to a concise, actionable message."""
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
    if "error" in out:
        return "Rust compilation failed. Use --verbose to see the full compiler output."
    return "Cargo build failed. Use --verbose to see the full output."


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
