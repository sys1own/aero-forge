"""Provider-agnostic LLM clients with retry and graceful auth handling."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from aero_forge.config import (
    DEFAULT_TIER_MODELS,
    ConfigOverride,
    Tier,
    current_override,
    resolve_tier_model,
)

logger = logging.getLogger("aero_forge.llm")


def _telemetry_dir() -> Path:
    """Return the directory used for LLM call telemetry."""
    path = Path(os.getenv("AERO_FORGE_TELEMETRY_DIR", "/tmp/aero-forge-telemetry"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _approx_token_count(text: str) -> int:
    """Approximate token count without requiring tiktoken."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _log_llm_telemetry(
    provider: Optional[str],
    model: Optional[str],
    prompt: Union[str, List[Dict[str, str]]],
    response: Optional[str],
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """Append a structured record of an LLM call to the telemetry log."""
    try:
        if isinstance(prompt, list):
            prompt_text = "\n".join(m.get("content", "") for m in prompt)
        else:
            prompt_text = str(prompt)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_token_count": _approx_token_count(prompt_text),
            "prompt": prompt_text,
            "response_preview": (response or "")[:2000] if not error else None,
            "response_empty": not bool(response),
            "error": error,
            **extra,
        }
        log_path = _telemetry_dir() / "llm_calls.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("Failed to write LLM telemetry: %s", exc)


class LLMError(Exception):
    """Raised when the LLM client cannot complete a request."""


def _normalize_messages(
    prompt: Union[str, List[Dict[str, str]]],
) -> List[Dict[str, str]]:
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
        model: Optional[str] = None,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
        tier: Union[str, Tier] = Tier.FAST,
        model_resolver: Optional[Callable[[str, str], Optional[str]]] = None,
    ):
        self.provider = provider
        self.tier = Tier(tier) if tier else Tier.FAST
        self._model_resolver = model_resolver
        self._model_is_pinned = model is not None
        self.model = model
        self.max_retries = max(1, max_retries)
        self.api_key = api_key
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.timeout = timeout or float(
            os.getenv("AERO_FORGE_LLM_TIMEOUT", "120.0")
        )
        # Resolve a concrete default model for the requested tier when no
        # explicit model is supplied. This makes ``client.model`` inspectable
        # while still honoring tier switches at ``generate()`` time.
        if not self._model_is_pinned:
            try:
                self.model = self._resolve_model(self.tier)
            except LLMError:
                self.model = None

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float = 0.2,
        *,
        tier: Optional[Union[str, Tier]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Generate a completion with exponential backoff retry.

        Honors server-provided retry delays (e.g. ``Retry-After`` headers or
        Google RPC ``retry_delay``) while still capping the wait at
        ``backoff_max``.
        """
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
            try:
                response = self._call(prompt, temperature, tier=tier, **kwargs)
                _log_llm_telemetry(
                    self.provider,
                    self.model,
                    prompt,
                    response,
                    attempt=attempt + 1,
                    temperature=temperature,
                    tier=str(tier) if tier is not None else None,
                )
                return response
            except LLMError:
                # Configuration or usage errors should not be retried.
                _log_llm_telemetry(
                    self.provider,
                    self.model,
                    prompt,
                    None,
                    error="LLMError",
                    attempt=attempt + 1,
                )
                raise
            except self._retryable_exceptions() as exc:
                logger.warning(
                    "Retryable error for %s (attempt %d): %s",
                    self.model,
                    attempt + 1,
                    exc,
                )
                _log_llm_telemetry(
                    self.provider,
                    self.model,
                    prompt,
                    None,
                    error=f"retryable: {exc}",
                    attempt=attempt + 1,
                )
                last_error = exc
                server_delay = self._extract_retry_delay(exc)
                if server_delay is not None:
                    delay = min(
                        max(server_delay, self.backoff_initial), self.backoff_max
                    )
                else:
                    delay = min(delay * 2, self.backoff_max)
                continue
            except Exception as exc:
                logger.error(
                    "Non-retryable error for %s (attempt %d): %s",
                    self.model,
                    attempt + 1,
                    exc,
                )
                _log_llm_telemetry(
                    self.provider,
                    self.model,
                    prompt,
                    None,
                    error=f"non-retryable: {exc}",
                    attempt=attempt + 1,
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

    @staticmethod
    def _extract_retry_delay(exc: Exception) -> Optional[float]:
        """Return the server-suggested retry delay in seconds, if any."""
        # OpenAI-compatible errors sometimes expose a parsed retry_after value.
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass

        # Generic HTTP response with Retry-After header.
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            for key in ("retry-after", "Retry-After"):
                if key in headers:
                    try:
                        return float(headers[key])
                    except (TypeError, ValueError):
                        pass

        # Google API exceptions may carry a retry_delay timedelta.
        retry_delay = getattr(exc, "retry_delay", None)
        if retry_delay is not None:
            try:
                return float(retry_delay.total_seconds())
            except (TypeError, AttributeError, ValueError):
                pass

        # Fallback: parse the textual repr for google.rpc.retryinfo blocks.
        try:
            text = str(exc)
            match = re.search(
                r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                return float(match.group(1))
        except (TypeError, ValueError):
            pass

        return None

    @abstractmethod
    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        *,
        tier: Optional[Union[str, Tier]] = None,
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

    def _resolve_model(
        self, tier: Optional[Union[str, Tier]] = None
    ) -> str:
        """Return the concrete model name for this request.

        An explicit model passed at construction time is always pinned. Otherwise
        the model is resolved from the configured provider + tier mapping.
        """
        if self._model_is_pinned and self.model:
            return self.model
        requested = Tier(tier) if tier else self.tier
        if self._model_resolver and self.provider:
            resolved = self._model_resolver(self.provider, requested.value)
            if resolved:
                return resolved
        # Fallback to the default model computed at construction time.
        if self.model:
            return self.model
        raise LLMError(
            f"No model configured for provider {self.provider!r} tier {requested.value!r}"
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible chat completion client."""

    def __init__(
        self,
        model: Optional[str] = None,
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
        *,
        tier: Optional[Union[str, Tier]] = None,
        **kwargs: Any,
    ) -> str:
        from openai import OpenAI

        api_key = self._resolve_key(["OPENAI_API_KEY", "AERO_FORGE_API_KEY"])
        if not api_key:
            raise LLMError(
                "OpenAI API key not found. Set OPENAI_API_KEY or AERO_FORGE_API_KEY."
            )

        base_url = (
            os.getenv("AERO_FORGE_BASE_URL")
            or self.base_url
            or "https://api.openai.com/v1"
        )
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        messages = _normalize_messages(prompt)
        response = client.chat.completions.create(
            model=self._resolve_model(tier),
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

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        super().__init__(model, **kwargs)

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        *,
        tier: Optional[Union[str, Tier]] = None,
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
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        messages = _normalize_messages(prompt)
        response = client.chat.completions.create(
            model=self._resolve_model(tier),
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content


class DeepSeekClient(OpenAIClient):
    """DeepSeek API uses an OpenAI-compatible endpoint."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        kwargs.setdefault("base_url", "https://api.deepseek.com/v1")
        super().__init__(model, **kwargs)

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        *,
        tier: Optional[Union[str, Tier]] = None,
        **kwargs: Any,
    ) -> str:
        api_key = self._resolve_key(["DEEPSEEK_API_KEY", "AERO_FORGE_API_KEY"])
        if not api_key:
            raise LLMError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY or AERO_FORGE_API_KEY."
            )
        self.api_key = api_key
        return super()._call(prompt, temperature, tier=tier, **kwargs)


class GeminiClient(BaseLLMClient):
    """Google Gemini client using the google-generativeai SDK."""

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float,
        *,
        tier: Optional[Union[str, Tier]] = None,
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
        model_name = self._resolve_model(tier)
        model = genai.GenerativeModel(model_name)
        messages = _normalize_messages(prompt)
        content = _messages_to_string(messages)

        generation_config = {"temperature": temperature}
        # Map OpenAI-style JSON mode to Gemini's response MIME type.
        response_format = kwargs.pop("response_format", None)
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"
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


_UNKNOWN_PROVIDER_FALLBACKS: Dict[str, str] = {
    "openai": "gpt-4",
    "openrouter": "openrouter/free",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
}


def get_llm_client(
    provider: Optional[str],
    model: Optional[str] = None,
    max_retries: int = 3,
    api_key: Optional[str] = None,
    config_override: Optional[ConfigOverride] = None,
    raise_on_error: bool = False,
    tier: Optional[Union[str, Tier]] = None,
    file_config: Optional[Dict[str, Any]] = None,
) -> Optional[BaseLLMClient]:
    """Return a configured LLM client for ``provider``.

    Returns ``None`` when provider is ``none``/empty or when a required key is
    missing, after logging a clear error. When ``raise_on_error`` is ``True``,
    missing API keys or unsupported providers raise ``LLMError`` instead of
    silently returning ``None``.

    ``tier`` selects the cost-capable model tier. If ``model`` is supplied
    explicitly it overrides the tier mapping. The requested ``tier`` is still
    honored on subsequent ``generate(tier=...)`` calls unless an explicit model
    was pinned at construction time.
    """
    override = config_override or current_override()
    if override is not None:
        if provider is None:
            provider = override.llm_provider
        if model is None:
            model = override.model
        if api_key is None:
            api_key = override.api_key
        if override.max_retries is not None:
            max_retries = override.max_retries

    # Treat empty/whitespace values as unset so they cannot shadow environment
    # variables or file-level defaults.
    if isinstance(provider, str):
        provider = provider.strip()
    if provider is not None and not provider:
        provider = None
    if isinstance(model, str):
        model = model.strip()
    if model is not None and not model:
        model = None
    if isinstance(api_key, str):
        api_key = api_key.strip()
    if api_key is not None and not api_key:
        api_key = None

    if not provider or provider.lower() in {"none", "null", ""}:
        return None

    tier_obj = Tier(tier) if tier else Tier.FAST

    if file_config is None:
        from aero_forge.config import find_config, load_config

        cfg_path = find_config()
        file_config = load_config(cfg_path) if cfg_path else {}

    # Tier-aware model resolution. An explicit ``model`` argument or the legacy
    # ``AERO_FORGE_MODEL`` environment variable pins the model. Otherwise the
    # provider's tier mapping is used, allowing ``generate(tier=...)`` to switch
    # between fast and reasoning models at call time.
    explicit_model = model or os.getenv("AERO_FORGE_MODEL")
    if explicit_model:
        model_name = explicit_model
    elif provider in DEFAULT_TIER_MODELS:
        model_name = None
    else:
        model_name = _UNKNOWN_PROVIDER_FALLBACKS.get(provider)

    model_resolver: Optional[Callable[[str, str], Optional[str]]]
    if provider in DEFAULT_TIER_MODELS:
        model_resolver = lambda p, t: resolve_tier_model(
            p, t, file_config=file_config, override=override
        )
    else:
        model_resolver = None

    def _key(*names: str) -> Optional[str]:
        if api_key:
            return api_key
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return None

    if provider == "openai":
        key = _key("OPENAI_API_KEY", "AERO_FORGE_API_KEY")
        if not key:
            msg = "OpenAI provider selected but OPENAI_API_KEY or AERO_FORGE_API_KEY is not set."
            logger.error(msg)
            if raise_on_error:
                raise LLMError(msg)
            return None
        return OpenAIClient(
            model=model_name,
            provider=provider,
            tier=tier_obj,
            model_resolver=model_resolver,
            max_retries=max_retries,
            api_key=key,
        )

    if provider == "openrouter":
        key = _key("OPENROUTER_API_KEY", "AERO_FORGE_API_KEY")
        if not key:
            msg = "OpenRouter provider selected but OPENROUTER_API_KEY or AERO_FORGE_API_KEY is not set."
            logger.error(msg)
            if raise_on_error:
                raise LLMError(msg)
            return None
        return OpenRouterClient(
            model=model_name,
            provider=provider,
            tier=tier_obj,
            model_resolver=model_resolver,
            max_retries=max_retries,
            api_key=key,
        )

    if provider == "deepseek":
        key = _key("DEEPSEEK_API_KEY", "AERO_FORGE_API_KEY")
        if not key:
            msg = "DeepSeek provider selected but DEEPSEEK_API_KEY or AERO_FORGE_API_KEY is not set."
            logger.error(msg)
            if raise_on_error:
                raise LLMError(msg)
            return None
        return DeepSeekClient(
            model=model_name,
            provider=provider,
            tier=tier_obj,
            model_resolver=model_resolver,
            max_retries=max_retries,
            api_key=key,
        )

    if provider == "gemini":
        if importlib.util.find_spec("google.generativeai") is None:
            raise ImportError(
                "Gemini provider requires the google-generativeai package. "
                "Install it with: pip install google-generativeai"
            )
        key = _key("GEMINI_API_KEY", "AERO_FORGE_API_KEY")
        if not key:
            msg = "Gemini provider selected but GEMINI_API_KEY or AERO_FORGE_API_KEY is not set."
            logger.error(msg)
            if raise_on_error:
                raise LLMError(msg)
            return None
        return GeminiClient(
            model=model_name,
            provider=provider,
            tier=tier_obj,
            model_resolver=model_resolver,
            max_retries=max_retries,
            api_key=key,
        )

    msg = (
        f"Unknown LLM provider: {provider}. "
        "Supported: openai, openrouter, deepseek, gemini, none."
    )
    logger.error(msg)
    if raise_on_error:
        raise LLMError(msg)
    return None
