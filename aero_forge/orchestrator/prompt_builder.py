"""Build consolidated LLM prompts from accumulated build/test errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Accumulate error context and produce a single repair prompt."""

    def __init__(self, system_message: Optional[str] = None):
        self.system_message = (
            system_message
            or "You are an expert Python and Rust programmer. Fix the provided function so it compiles and passes its tests. If the code already compiles but tests fail, correct the algorithm based on the test output. Return ONLY the corrected function definition (no markdown fences, no explanation)."
        )
        self.errors: List[str] = []

    def add_error(self, error: str) -> None:
        """Add an error message to the context."""
        if error and error not in self.errors:
            self.errors.append(error)

    def clear(self) -> None:
        self.errors.clear()

    def build(
        self,
        function_name: str,
        function_source: str,
        additional_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of chat-completion messages."""
        parts = [
            f"Fix the Python function `{function_name}` so that it compiles and passes its tests.",
            "",
            f"Function `{function_name}`:",
            function_source,
        ]
        if self.errors:
            parts.extend(["", "Accumulated failures:"])
            for idx, err in enumerate(self.errors, 1):
                parts.append(f"[{idx}] {err}")
        if additional_context:
            parts.extend(["", "Additional context:", additional_context])
        parts.extend(["", "Return ONLY the corrected function definition."])
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": "\n".join(parts)},
        ]


__all__ = ["PromptBuilder"]
