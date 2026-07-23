"""Multi-provider LLM client with retry, backoff, and model fallback."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aero_forge.model_client")


def _detect_provider(model: str) -> Tuple[str, str]:
    """Return (base_url, actual_model_name) for ``model``."""
    model = model.strip()
    if model.startswith("openrouter/"):
        name = model.split("/", 1)[1] or "openai/gpt-4"
        return ("https://openrouter.ai/api/v1", name)
    if model.startswith("gemini-") or "/gemini" in model:
        return ("https://generativelanguage.googleapis.com/v1beta/openai/", model)
    return ("https://api.openai.com/v1", model)


class ModelClient:
    """Call a prioritized list of OpenAI-compatible chat-completion endpoints."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        max_retries: int = 3,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        api_key: Optional[str] = None,
    ):
        self.models = list(models) if models else ["gpt-4"]
        self.max_retries = max(1, max_retries)
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.api_key = api_key or os.getenv("AERO_FORGE_API_KEY")

    def _client_for(self, model: str) -> Tuple[Any, str]:
        from openai import OpenAI

        base_url, actual_model = _detect_provider(model)
        base_url = os.getenv("AERO_FORGE_BASE_URL") or base_url
        api_key = self.api_key
        if not api_key:
            # Some local endpoints don't require a key.
            api_key = "dummy-key"
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
        return client, actual_model

    def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Optional[str]:
        """Return the first successful completion content, or ``None`` if all fail."""
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )

        for model in self.models:
            try:
                client, actual_model = self._client_for(model)
            except Exception as exc:
                logger.warning("Could not create client for %s: %s", model, exc)
                continue

            delay = self.backoff_initial
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries):
                if attempt > 0:
                    logger.info("Retrying %s in %.1fs (attempt %d)", model, delay, attempt + 1)
                    time.sleep(delay)
                    delay = min(delay * 2, self.backoff_max)

                try:
                    response = client.chat.completions.create(
                        model=actual_model,
                        messages=messages,
                        temperature=temperature,
                        **kwargs,
                    )
                    content = response.choices[0].message.content
                    logger.info("Success with model %s", model)
                    return content
                except RateLimitError as exc:
                    logger.warning("Rate limit on %s: %s", model, exc)
                    last_error = exc
                    # Rate limits are transient; retry/backoff.
                    continue
                except (APIConnectionError, APITimeoutError, APIError) as exc:
                    logger.warning("Transient API error on %s: %s", model, exc)
                    last_error = exc
                    continue
                except AuthenticationError as exc:
                    logger.warning("Authentication failed for %s: %s", model, exc)
                    # Don't retry the same model with a bad/expired key.
                    break
                except Exception as exc:
                    logger.warning("Non-retryable error with %s: %s", model, exc)
                    break

            if last_error:
                logger.warning("Model %s failed after retries: %s", model, last_error)

        logger.error("All models failed to produce a completion.")
        return None


__all__ = ["ModelClient"]
