"""Provider-agnostic LLM clients with retry and graceful auth handling."""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("aero_forge.llm")


class LLMError(Exception):
    """Raised when the LLM client cannot complete a request."""


def _normalize_messages(prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages = []
    for message in prompt:
        if isinstance(message, dict):
            messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                }
            )
    return messages


def _messages_to_string(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            parts.append(f"System instruction:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


class BaseLLMClient(ABC):
    """Abstract base for an LLM provider client."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self.model = model
        self.max_retries = max(1, max_retries)
        self.api_key = api_key
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Optional[str]:
        """Generate a completion with exponential backoff retry."""
        delay = self.backoff_initial
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                logger.info(
                    "Retrying %s in %.1fs (attempt %d/%d)",
                    self.model,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
                delay = min(delay * 2, self.backoff_max)
            try:
                return self._call(prompt, temperature, **kwargs)
            except LLMError:
                # Configuration or usage errors should not be retried.
                raise
            except self._retryable_exceptions() as exc:
                logger.warning(
                    "Retryable error for %s (attempt %d): %s",
                    self.model,
                    attempt + 1,
                    exc,
                )
                last_error = exc
                continue
            except Exception as exc:
                logger.error(
                    "Non-retryable error for %s (attempt %d): %s",
                    self.model,
                    attempt + 1,
                    exc,
                )
                break

        if last_error:
            logger.error(
                "LLM %s failed after %d retries: %s",
                self.model,
                self.max_retries,
                last_error,
            )
        return None

    @abstractmethod
    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Provider-specific completion call."""

    @abstractmethod
    def _retryable_exceptions(self) -> Any:
        """Return a tuple of exceptions that should trigger a retry."""

    def _resolve_key(self, env_names: List[str]) -> Optional[str]:
        if self.api_key:
            return self.api_key
        for name in env_names:
            key = os.getenv(name)
            if key:
                return key
        return None


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible chat completion client."""

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        from openai import OpenAI

        api_key = self._resolve_key(["OPENAI_API_KEY", "AERO_FORGE_API_KEY"])
        if not api_key:
            raise LLMError(
                "OpenAI API key not found. Set OPENAI_API_KEY or AERO_FORGE_API_KEY."
            )

        base_url = os.getenv("AERO_FORGE_BASE_URL") or self.base_url or "https://api.openai.com/v1"
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        messages = _normalize_messages(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    def _retryable_exceptions(self) -> Any:
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

        return (APIConnectionError, APIError, APITimeoutError, RateLimitError)


class OpenRouterClient(OpenAIClient):
    """OpenRouter uses an OpenAI-compatible endpoint with its own defaults."""

    def __init__(self, model: str, **kwargs: Any):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        super().__init__(model, **kwargs)

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        api_key = self._resolve_key(["OPENROUTER_API_KEY", "AERO_FORGE_API_KEY"])
        if not api_key:
            raise LLMError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY or AERO_FORGE_API_KEY."
            )
        # Re-use OpenAI-compatible machinery with the OpenRouter base URL and key.
        from openai import OpenAI

        base_url = os.getenv("AERO_FORGE_BASE_URL") or self.base_url
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        messages = _normalize_messages(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content


class GeminiClient(BaseLLMClient):
    """Google Gemini client using the google-generativeai SDK."""

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        import importlib

        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError as exc:
            raise LLMError(
                "Gemini provider requires the google-generativeai package. "
                "Install it with: pip install google-generativeai"
            ) from exc

        api_key = self._resolve_key(["GEMINI_API_KEY", "AERO_FORGE_API_KEY"])
        if not api_key:
            raise LLMError(
                "Gemini API key not found. Set GEMINI_API_KEY or AERO_FORGE_API_KEY."
            )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        messages = _normalize_messages(prompt)
        content = _messages_to_string(messages)

        generation_config = {"temperature": temperature}
        # Allow passing additional gemini kwargs through, but keep config separate.
        gemini_kwargs = {
            "generation_config": generation_config,
            **kwargs,
        }
        response = model.generate_content(content, **gemini_kwargs)
        return response.text

    def _retryable_exceptions(self) -> Any:
        try:
            from google.api_core import exceptions as google_exceptions

            return (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded,
                google_exceptions.InternalServerError,
            )
        except ImportError:
            return (Exception,)


def get_llm_client(
    provider: Optional[str],
    model: Optional[str] = None,
    max_retries: int = 3,
    api_key: Optional[str] = None,
) -> Optional[BaseLLMClient]:
    """Return a configured LLM client for ``provider``.

    Returns ``None`` when provider is ``none``/empty or when a required key is
    missing, after logging a clear error.
    """
    if not provider or provider.lower() in {"none", "null", ""}:
        return None

    provider = provider.lower()

    if provider == "openai":
        resolved_model = model or os.getenv("AERO_FORGE_MODEL") or "gpt-4"
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AERO_FORGE_API_KEY")
        if not key:
            logger.error(
                "OpenAI provider selected but OPENAI_API_KEY or AERO_FORGE_API_KEY is not set."
            )
            return None
        return OpenAIClient(model=resolved_model, max_retries=max_retries, api_key=key)

    if provider == "openrouter":
        resolved_model = model or os.getenv("AERO_FORGE_MODEL") or "openrouter/free"
        key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("AERO_FORGE_API_KEY")
        )
        if not key:
            logger.error(
                "OpenRouter provider selected but OPENROUTER_API_KEY or AERO_FORGE_API_KEY is not set."
            )
            return None
        return OpenRouterClient(
            model=resolved_model, max_retries=max_retries, api_key=key
        )

    if provider == "gemini":
        resolved_model = model or os.getenv("AERO_FORGE_MODEL") or "gemini-2.0-flash"
        if importlib.util.find_spec("google.generativeai") is None:
            raise ImportError(
                "Gemini provider requires the google-generativeai package. "
                "Install it with: pip install google-generativeai"
            )
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("AERO_FORGE_API_KEY")
        if not key:
            logger.error(
                "Gemini provider selected but GEMINI_API_KEY or AERO_FORGE_API_KEY is not set."
            )
            return None
        return GeminiClient(model=resolved_model, max_retries=max_retries, api_key=key)

    logger.error("Unknown LLM provider: %s. Supported: openai, openrouter, gemini, none.", provider)
    return None
