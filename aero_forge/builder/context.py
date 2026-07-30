"""Workspace context builder for follow-up build feedback loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from aero_forge.bundle_repo import bundle_workspace, format_context_block
from aero_forge.healing.context_builder import ContextBuilder


class WorkspaceContext:
    """Provide bundled workspace context for builder feedback and LLM prompts."""

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path).resolve()

    def bundle(self, max_file_size_kb: int = 100) -> Dict[str, Any]:
        """Return the workspace file bundle."""
        return bundle_workspace(self.workspace_path, max_file_size_kb=max_file_size_kb)

    def format(self, fmt: str = "xml", max_file_size_kb: int = 100) -> str:
        """Return a serialized workspace block suitable for injection into prompts."""
        return format_context_block(self.bundle(max_file_size_kb=max_file_size_kb), fmt=fmt)

    def failure_context(
        self,
        command: str,
        exit_code: int,
        log_text: str,
        diagnosis: Optional[Dict[str, Any]] = None,
        previous_attempts: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Build a structured failure context from the current workspace."""
        return ContextBuilder(self.workspace_path).build_failure_context(
            command, exit_code, log_text, diagnosis, previous_attempts or []
        )


def get_workspace_context(workspace_path: Path) -> WorkspaceContext:
    """Return a context object for *workspace_path*."""
    return WorkspaceContext(workspace_path)
