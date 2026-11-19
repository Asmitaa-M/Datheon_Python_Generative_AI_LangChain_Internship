"""LLM integration (kept separate from the extraction business logic).

* :class:`BaseLLMClient`  - the tiny interface the extractor depends on.
* :class:`GeminiClient`   - real calls through the ``google-genai`` SDK.
* :func:`create_llm_client` - picks the mock or the real client from settings.

Provider errors are translated into the project's own exception types so the
rest of the code never has to know about SDK internals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from config import Settings
from exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    MissingAPIKeyError,
)
from utils import redact_secrets

logger = logging.getLogger(__name__)

_MAX_ERROR_MESSAGE_CHARS = 300


class BaseLLMClient(ABC):
    """Anything that can turn a prompt into response text."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send ``prompt`` to the model and return its raw text response.

        Raises:
            LLMError: (or a subclass) when the call fails.
        """


class GeminiClient(BaseLLMClient):
    """Google Gemini through the current ``google-genai`` SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise MissingAPIKeyError("An API key is required to call Gemini.")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        # The SDK expects the timeout in milliseconds.
        http_options = genai_types.HttpOptions(timeout=int(timeout_seconds * 1000))
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def generate(self, prompt: str) -> str:
        config = genai_types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json",
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        except genai_errors.APIError as exc:
            raise self._translate_api_error(exc) from exc
        except Exception as exc:
            if _is_connection_failure(exc):
                message = self._safe_message(exc)
                raise LLMConnectionError(
                    f"Could not reach the Gemini API: {message}"
                ) from exc
            raise  # unexpected: never swallow it

        text = response.text
        if not text or not text.strip():
            raise LLMError(
                "Gemini returned an empty response "
                "(it may have been blocked by a safety filter)."
            )
        return text

    # -- helpers ----------------------------------------------------------- #
    def _safe_message(self, exc: BaseException) -> str:
        text = redact_secrets(str(exc), secrets=(self._api_key,))
        return text[:_MAX_ERROR_MESSAGE_CHARS]

    def _translate_api_error(self, exc: genai_errors.APIError) -> LLMError:
        code: Optional[int] = getattr(exc, "code", None)
        detail = redact_secrets(
            str(getattr(exc, "message", "") or exc), secrets=(self._api_key,)
        )[:_MAX_ERROR_MESSAGE_CHARS]

        if code == 429:
            return LLMRateLimitError(
                f"Gemini rate limit or quota reached (HTTP 429): {detail}"
            )
        if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
            return LLMAuthenticationError(
                "Gemini rejected the API key. Check GEMINI_API_KEY in your "
                f".env file (HTTP {code}): {detail}"
            )
        if isinstance(code, int) and code >= 500:
            return LLMServerError(
                f"Gemini service is temporarily unavailable (HTTP {code}): {detail}"
            )
        return LLMError(f"Gemini API request failed (HTTP {code}): {detail}")


def _is_connection_failure(exc: BaseException) -> bool:
    """Detect network failures without depending on a specific HTTP library.

    ``google-genai`` uses ``httpx``-style transports whose exceptions derive
    from a class named ``TransportError``; plain sockets raise ``OSError``.
    """
    if isinstance(exc, (OSError, TimeoutError)):
        return True
    return any(
        cls.__name__ == "TransportError" and cls.__module__.startswith("httpx")
        for cls in type(exc).__mro__
    )


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """Create the right client for the current settings.

    Raises:
        MissingAPIKeyError: If mock mode is off and no usable key is set.
    """
    if settings.mock_mode:
        from mock_llm import MockLLMClient  # local import: only needed in mock mode

        logger.info("Mock mode is ON: no API calls will be made.")
        return MockLLMClient()

    if not settings.has_api_key or settings.api_key is None:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example), or run with --mock to test without an API key."
        )
    logger.info("Using Gemini model %r.", settings.model)
    return GeminiClient(
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
    )
