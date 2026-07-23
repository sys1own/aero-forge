"""Interactive chat session for prompt-driven code generation and optimization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.error_explainer import explain_error
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
        prompt_template: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.llm_provider = llm_provider
        self.model = model
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.prompt_template = prompt_template
        self.messages: List[Dict[str, str]] = []
        self.last_error: Optional[str] = None
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
        lowered = text.lower().strip()
        if lowered in ("help", "?"):
            return self._help_action()
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

        if "benchmark" in lowered or "speed" in lowered:
            return self._benchmark_action(text)

        if any(
            phrase in lowered for phrase in ("build", "compile", "run tests", "test")
        ):
            return self._build_action(text)

        if any(phrase in lowered for phrase in ("show", "display", "view code")):
            return self._show_action()

        if any(phrase in lowered for phrase in ("explain", "why")):
            return self._explain_action()

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
            prompt_template=self.prompt_template,
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
                prompt_template=self.prompt_template,
            ),
        }

    def _build_action(self, text: str) -> Dict[str, Any]:
        """Compile the current generated project."""
        result = generate_and_build(
            text,
            output_dir=self.output_dir,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            prompt_template=self.prompt_template,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
        )
        build = result.get("build") or {}
        if not build.get("success"):
            self.last_error = str(build)
        return result

    def _benchmark_action(self, text: str) -> Dict[str, Any]:
        """Build and time the current generated project."""
        import time

        start = time.perf_counter()
        result = self._build_action(text)
        elapsed = time.perf_counter() - start
        build = result.get("build") or {}
        result["benchmark_seconds"] = elapsed
        result["message"] = (
            f"Benchmark: {elapsed:.3f}s build time"
            if build.get("success")
            else f"Build failed after {elapsed:.3f}s"
        )
        return result

    def _show_action(self) -> Dict[str, Any]:
        """Return the contents of the most recently generated source file."""
        source_path = self.output_dir / "src" / "generated.py"
        if not source_path.exists():
            return {"message": "No generated code yet. Try 'generate ...' first."}
        return {
            "message": f"Generated source ({source_path}):\n\n"
            + source_path.read_text(encoding="utf-8")
        }

    def _explain_action(self) -> Dict[str, Any]:
        """Explain the last build error."""
        if not self.last_error:
            return {"message": "No error to explain. Build something first."}
        source_path = self.output_dir / "src" / "generated.py"
        source = (
            source_path.read_text(encoding="utf-8") if source_path.exists() else None
        )
        explanation = explain_error(
            self.last_error,
            source=source,
            llm_provider=self.llm_provider,
            model=self.model,
        )
        return {"message": explanation}

    def _help_action(self) -> Dict[str, Any]:
        """Return a help message for chat commands."""
        return {
            "message": (
                "Available commands:\n"
                "  generate <prompt>  – create code from a prompt\n"
                "  build              – compile the current project\n"
                "  test               – run tests for the current project\n"
                "  optimize <prompt>  – optimize the current code\n"
                "  benchmark          – build and time the project\n"
                "  show               – display generated source\n"
                "  explain            – explain the last build error\n"
                "  help               – show this help\n"
                "  exit / quit        – leave chat"
            )
        }
