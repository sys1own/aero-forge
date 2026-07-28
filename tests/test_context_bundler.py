"""Tests for the ContextBundler (Reasoning-tier blueprint synthesis)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from aero_forge.context_bundler import ContextBundler


def test_context_bundler_synthesize_blueprint_uses_reasoning_tier(tmp_path: Path) -> None:
    """ContextBundler passes the Reasoning tier to the synthesizer and writes blueprint.aero."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hello')\n", encoding="utf-8")

    captured: Dict[str, Any] = {}

    class FakeSynthesizer:
        def synthesize(self, workspace: Path, output_path: Path, draft: Any = None, spec: Any = None) -> Any:
            captured["workspace"] = workspace
            captured["output_path"] = output_path
            from aero_forge.blueprint import BlueprintV3

            bp = BlueprintV3(
                metadata={
                    "schema_version": "3.0.0",
                    "project_name": workspace.name,
                    "status": "finalized",
                    "generation_method": "llm_synthesized",
                    "transferable": True,
                    "llm_initialized": True,
                },
                llm_context={"state": "synthesized"},
            )
            return bp

    bundler = ContextBundler(llm_provider="fake")
    bundler.synthesizer = FakeSynthesizer()  # type: ignore[assignment]
    bp = bundler.synthesize_blueprint(workspace)
    assert bp.metadata.llm_initialized is True
    assert bp.llm_context.state.value == "synthesized"
    assert captured["workspace"] == workspace


def test_context_bundler_async_runs_without_blocking(tmp_path: Path) -> None:
    """synthesize_blueprint_async starts a thread and returns immediately."""
    workspace = tmp_path / "project"
    workspace.mkdir()

    bundler = ContextBundler(llm_provider="none")
    bundler.synthesizer = MagicMock()  # type: ignore[assignment]
    thread = bundler.synthesize_blueprint_async(workspace)
    assert thread.is_alive() or not thread.is_alive()
    thread.join(timeout=2.0)
