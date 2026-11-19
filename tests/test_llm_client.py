"""Tests for the Gemini client wrapper. The SDK itself is always mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

import llm_client
from config import Settings, load_settings
from exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    MissingAPIKeyError,
)
from llm_client import GeminiClient, create_llm_client
from mock_llm import MockLLMClient

SECRET = "AIzaSyD-NOT_A_REAL_KEY_but_looks_like_one_123"


@pytest.fixture
def sdk(monkeypatch):
    """Replace genai.Client with a mock and return the mocked ``generate_content``."""
    client_class = MagicMock(name="genai.Client")
    monkeypatch.setattr(llm_client.genai, "Client", client_class)
    client_class.generate = client_class.return_value.models.generate_content
    return client_class


def api_error(code: int, message: str, status: str = "ERROR"):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"message": message, "status": status, "code": code}})


def make_client() -> GeminiClient:
    return GeminiClient(api_key=SECRET, model="gemini-test", temperature=0.0, timeout_seconds=12)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_missing_api_key_raises_a_helpful_error():
    with pytest.raises(MissingAPIKeyError) as caught:
        create_llm_client(load_settings({}))
    message = str(caught.value)
    assert "GEMINI_API_KEY" in message and ".env" in message and "--mock" in message


@pytest.mark.parametrize("placeholder", ["", "your_api_key_here"])
def test_placeholder_api_key_counts_as_missing(placeholder):
    with pytest.raises(MissingAPIKeyError):
        create_llm_client(load_settings({"GEMINI_API_KEY": placeholder}))


def test_mock_mode_needs_no_api_key_and_makes_no_sdk_client(sdk):
    client = create_llm_client(Settings(mock_mode=True))
    assert isinstance(client, MockLLMClient)
    sdk.assert_not_called()


def test_real_client_is_created_from_settings(sdk):
    client = create_llm_client(
        load_settings({"GEMINI_API_KEY": SECRET, "GEMINI_MODEL": "gemini-x", "LLM_TIMEOUT_SECONDS": "7"})
    )
    assert isinstance(client, GeminiClient)
    kwargs = sdk.call_args.kwargs
    assert kwargs["api_key"] == SECRET
    assert kwargs["http_options"].timeout == 7000  # milliseconds


def test_gemini_client_requires_a_key():
    with pytest.raises(MissingAPIKeyError):
        GeminiClient(api_key="", model="m")


# --------------------------------------------------------------------------- #
# Successful call
# --------------------------------------------------------------------------- #
def test_generate_returns_response_text_and_sends_expected_request(sdk):
    sdk.generate.return_value = SimpleNamespace(text='{"ok": true}')
    assert make_client().generate("my prompt") == '{"ok": true}'

    kwargs = sdk.generate.call_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"] == "my prompt"
    assert kwargs["config"].temperature == 0.0
    assert kwargs["config"].response_mime_type == "application/json"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_response_raises_llm_error(sdk, empty):
    sdk.generate.return_value = SimpleNamespace(text=empty)
    with pytest.raises(LLMError, match="empty response"):
        make_client().generate("p")


# --------------------------------------------------------------------------- #
# API failure handling
# --------------------------------------------------------------------------- #
def test_rate_limit_becomes_llm_rate_limit_error(sdk):
    sdk.generate.side_effect = api_error(429, "Quota exceeded", "RESOURCE_EXHAUSTED")
    with pytest.raises(LLMRateLimitError, match="rate limit"):
        make_client().generate("p")


@pytest.mark.parametrize(
    "code,message",
    [(401, "Unauthorized"), (403, "Permission denied"), (400, "API key not valid. Please pass a valid API key.")],
)
def test_bad_credentials_become_authentication_errors(sdk, code, message):
    sdk.generate.side_effect = api_error(code, message)
    with pytest.raises(LLMAuthenticationError, match="GEMINI_API_KEY"):
        make_client().generate("p")


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_server_errors_are_flagged_as_temporary(sdk, code):
    sdk.generate.side_effect = api_error(code, "overloaded", "UNAVAILABLE")
    with pytest.raises(LLMServerError):
        make_client().generate("p")


def test_other_client_errors_are_plain_llm_errors(sdk):
    sdk.generate.side_effect = api_error(404, "model not found")
    with pytest.raises(LLMError) as caught:
        make_client().generate("p")
    assert type(caught.value) is LLMError  # not retried


@pytest.mark.parametrize("failure", [ConnectionError("refused"), TimeoutError("slow"), OSError("network down")])
def test_connection_failures_become_connection_errors(sdk, failure):
    sdk.generate.side_effect = failure
    with pytest.raises(LLMConnectionError, match="Could not reach"):
        make_client().generate("p")


def test_transport_errors_from_the_http_library_are_recognised(sdk):
    class TransportError(Exception):
        """Mimics httpx.TransportError without importing it."""

    TransportError.__module__ = "httpx2._exceptions"
    sdk.generate.side_effect = TransportError("connect timeout")
    with pytest.raises(LLMConnectionError):
        make_client().generate("p")


def test_unexpected_exceptions_are_not_swallowed(sdk):
    sdk.generate.side_effect = KeyError("bug in the SDK")
    with pytest.raises(KeyError):
        make_client().generate("p")


@pytest.mark.parametrize(
    "side_effect",
    [
        api_error(400, f"API key not valid: {SECRET}"),
        api_error(429, f"quota exceeded for key={SECRET}"),
        api_error(503, f"unavailable (key {SECRET})"),
        api_error(404, f"not found {SECRET}"),
        ConnectionError(f"could not connect using {SECRET}"),
    ],
)
def test_api_key_never_appears_in_raised_errors(sdk, side_effect):
    sdk.generate.side_effect = side_effect
    with pytest.raises(LLMError) as caught:
        make_client().generate("p")
    assert SECRET not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_very_long_provider_messages_are_shortened(sdk):
    sdk.generate.side_effect = api_error(500, "x" * 5000)
    with pytest.raises(LLMServerError) as caught:
        make_client().generate("p")
    assert len(str(caught.value)) < 500
