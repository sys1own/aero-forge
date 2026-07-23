"""Interactive chat session for prompt-driven code generation and optimization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.generate import generate_and_build, optimize_generated_code
from aero_forge.llm.clients import get_llm_client

logger = logging.getLogger("aero_forge.chat")


class ChatSession:
    """Maintain multi-turn conversation state and dispatch code actions."""

    def __init__(
        self,
        output_dir: Path,
        *,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
        max_retries: int = 3,
    ):
        self.output_dir = output_dir
        self.llm_provider = llm_provider
        self.model = model
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = (
            "You are an expert Python/Rust engineer embedded in Aero-Forge, "
            "a transpiler that turns Python into native extensions. "
            "You help the user write, build, and optimize Python functions. "
            "When asked to produce or modify code, return the full implementation "
            "in a single Python fenced code block and, if tests are requested, "
            "a second fenced code block for pytest tests."
        )

    def reply(self, text: str) -> str:
        """Append user message, call the LLM, and return the assistant response."""
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.messages.append({"role": "user", "content": text})

        client = get_llm_client(
            self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
        )
        if client is None:
            return (
                "No LLM provider is configured. Set AERO_FORGE_LLM_PROVIDER "
                "or pass --llm-provider."
            )

        response = client.generate(self.messages, temperature=0.2)
        if not response:
            return "LLM returned an empty response."

        self.messages.append({"role": "assistant", "content": response})
        return response

    def handle_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect action verbs and optionally execute a build/optimize step.

        Returns a result dict when an action was taken, otherwise None.
        """
        lowered = text.lower()
        if any(
            phrase in lowered
            for phrase in ("generate", "write a function", "create", "implement")
        ):
            return self._generate_action(text)

        if any(
            phrase in lowered
            for phrase in ("optimize", "make it faster", "speed up", "faster")
        ):
            return self._optimize_action(text)

        if any(
            phrase in lowered for phrase in ("build", "compile", "run tests", "test")
        ):
            return self._build_action(text)

        return None

    def _generate_action(self, text: str) -> Dict[str, Any]:
        """Generate code from the most recent user message."""
        return generate_and_build(
            text,
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            max_iterations=self.max_iterations,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
        )

    def _optimize_action(self, text: str) -> Dict[str, Any]:
        """Run the iterative optimization loop on the current project."""
        return {
            "iterations": optimize_generated_code(
                output_dir=self.output_dir,
                prompt=text,
                llm_provider=self.llm_provider,
                model=self.model,
                max_retries=self.max_retries,
                max_iterations=self.max_iterations,
            ),
        }

    def _build_action(self, text: str) -> Dict[str, Any]:
        """Compile the current generated project."""
        return generate_and_build(
            text,
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
        )
