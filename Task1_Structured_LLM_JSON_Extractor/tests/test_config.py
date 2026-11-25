"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from config import DEFAULT_MODEL, Settings, load_settings
from exceptions import ConfigurationError


def test_defaults_when_nothing_is_configured():
    settings = load_settings({})
    assert settings.api_key is None
    assert settings.has_api_key is False
    assert settings.model == DEFAULT_MODEL
    assert settings.mock_mode is False
    assert settings.max_retries == 3
    assert settings.api_max_retries == 2
    assert settings.temperature == 0.0
    assert settings.output_dir == config.PROJECT_ROOT / "output"


def test_all_values_can_be_overridden_through_the_environment():
    settings = load_settings(
        {
            "GEMINI_API_KEY": "real-looking-key-123",
            "GEMINI_MODEL": "gemini-custom",
            "LLM_MOCK_MODE": "true",
            "MAX_RETRIES": "5",
            "API_MAX_RETRIES": "1",
            "API_RETRY_DELAY": "0.5",
            "LLM_TEMPERATURE": "0.3",
            "LLM_TIMEOUT_SECONDS": "30",
            "OUTPUT_DIR": "results",
        }
    )
    assert settings.api_key == "real-looking-key-123" and settings.has_api_key
    assert settings.model == "gemini-custom"
    assert settings.mock_mode is True
    assert (settings.max_retries, settings.api_max_retries) == (5, 1)
    assert settings.api_retry_delay == 0.5
    assert settings.temperature == 0.3
    assert settings.timeout_seconds == 30
    assert settings.output_dir == Path("results")


@pytest.mark.parametrize("value", ["", "   ", "your_api_key_here", "YOUR_API_KEY_HERE", "changeme"])
def test_blank_and_placeholder_keys_do_not_count_as_configured(value):
    assert load_settings({"GEMINI_API_KEY": value}).has_api_key is False


def test_api_key_is_hidden_from_repr_and_str():
    settings = load_settings({"GEMINI_API_KEY": "super-secret-key-value"})
    assert "super-secret-key-value" not in repr(settings)
    assert "super-secret-key-value" not in str(settings)


@pytest.mark.parametrize("truthy", ["true", "TRUE", "True", "1", "yes", "on", " true "])
def test_mock_mode_accepts_truthy_values(truthy):
    assert load_settings({"LLM_MOCK_MODE": truthy}).mock_mode is True


@pytest.mark.parametrize("falsy", ["false", "FALSE", "0", "no", "off", ""])
def test_mock_mode_accepts_falsy_values(falsy):
    assert load_settings({"LLM_MOCK_MODE": falsy}).mock_mode is False


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_MOCK_MODE", "maybe"),
        ("MAX_RETRIES", "abc"),
        ("MAX_RETRIES", "-1"),
        ("MAX_RETRIES", "11"),
        ("MAX_RETRIES", "2.5"),
        ("API_MAX_RETRIES", "-3"),
        ("API_RETRY_DELAY", "-1"),
        ("API_RETRY_DELAY", "soon"),
        ("LLM_TEMPERATURE", "hot"),
        ("LLM_TEMPERATURE", "5"),
        ("LLM_TIMEOUT_SECONDS", "0"),
    ],
)
def test_invalid_values_raise_a_clear_error_naming_the_setting(name, value):
    with pytest.raises(ConfigurationError, match=name):
        load_settings({name: value})


def test_dotenv_file_is_loaded(tmp_path, monkeypatch):
    fake_environ: dict[str, str] = {}
    monkeypatch.setattr(config.os, "environ", fake_environ)
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=from-dotenv-file\nGEMINI_MODEL=model-from-file\n", encoding="utf-8")

    settings = load_settings(dotenv_path=dotenv)
    assert settings.api_key == "from-dotenv-file"
    assert settings.model == "model-from-file"


def test_real_environment_variables_win_over_the_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config.os, "environ", {"GEMINI_MODEL": "from-environment"})
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_MODEL=from-file\n", encoding="utf-8")
    assert load_settings(dotenv_path=dotenv).model == "from-environment"


def test_missing_dotenv_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config.os, "environ", {})
    assert load_settings(dotenv_path=tmp_path / "nope.env").api_key is None


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(Exception):
        settings.model = "other"  # type: ignore[misc]
