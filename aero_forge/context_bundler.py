"""Context bundler (Reasoning Tier) that synthesizes a deep `blueprint.aero`.

The Context Bundler wraps `LLMBlueprintSynthesizer` and is invoked whenever the
Builder completes a workspace update or when the Copilot detects that the
workspace context is missing, stale, or not yet LLM-initialized.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from aero_forge.blueprint import BlueprintV3
from aero_forge.blueprint.synthesizer import LLMBlueprintSynthesizer

logger = logging.getLogger("aero_forge.context_bundler")


class ContextBundler:
    """Bundle a workspace and synthesize a rich Blueprint v3 using the Reasoning tier."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_override: Optional[Any] = None,
    ) -> None:
        self.synthesizer = LLMBlueprintSynthesizer(
            provider=llm_provider,
            model=model,
            api_key=api_key,
            config_override=config_override,
        )

    def synthesize_blueprint(
        self,
        workspace: Path,
        output_path: Optional[Path] = None,
    ) -> BlueprintV3:
        """Synthesize (or re-synthesize) `blueprint.aero` for *workspace*."""
        workspace = Path(workspace).resolve()
        if output_path is None:
            output_path = workspace / "blueprint.aero"
        return self.synthesizer.synthesize(
            workspace,
            output_path=output_path,
        )

    def synthesize_blueprint_async(
        self,
        workspace: Path,
        output_path: Optional[Path] = None,
    ) -> threading.Thread:
        """Start a non-blocking background thread to synthesize `blueprint.aero`.

        Returns the started thread so callers can join it if they need to wait.
        """

        def _run() -> None:
            try:
                self.synthesize_blueprint(workspace, output_path=output_path)
            except Exception as exc:
                logger.warning(
                    "Async blueprint synthesis failed for %s: %s", workspace, exc
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
