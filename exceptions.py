"""Custom exceptions used across the extractor.

Every error the application raises on purpose derives from
:class:`ExtractorError`, so callers (such as the CLI) can catch one base class
and still tell the specific failures apart.
"""

from __future__ import annotations

from typing import Sequence


class ExtractorError(Exception):
    """Base class for all expected, user-facing errors."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(ExtractorError):
    """Raised when settings are missing or invalid."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when a real LLM call is requested but no API key is configured."""


# --------------------------------------------------------------------------- #
# Input / output
# --------------------------------------------------------------------------- #
class UnsupportedDocumentTypeError(ExtractorError):
    """Raised when a document type other than invoice/resume is requested."""


class EmptyInputError(ExtractorError):
    """Raised when the document text is empty or whitespace only."""


class InputFileError(ExtractorError):
    """Raised when an input file is missing or cannot be read."""


class OutputFileError(ExtractorError):
    """Raised when a result file cannot be written."""


# --------------------------------------------------------------------------- #
# LLM / API
# --------------------------------------------------------------------------- #
class LLMError(ExtractorError):
    """Raised for any failure while talking to the LLM provider."""


class LLMAuthenticationError(LLMError):
    """Raised when the provider rejects the API key. Never retried."""


class LLMTransientError(LLMError):
    """Base class for temporary failures that are worth retrying."""


class LLMRateLimitError(LLMTransientError):
    """Raised when the provider reports that a rate limit or quota was hit."""


class LLMConnectionError(LLMTransientError):
    """Raised when the provider cannot be reached (network down, timeout)."""


class LLMServerError(LLMTransientError):
    """Raised when the provider reports a temporary server-side failure."""


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #
class ResponseValidationError(ExtractorError):
    """Raised when an LLM response is unusable.

    Attributes:
        errors: Human-readable problems, suitable for sending back to the LLM.
    """

    def __init__(self, message: str, errors: Sequence[str]) -> None:
        super().__init__(message)
        self.errors: list[str] = list(errors)


class InvalidJSONError(ResponseValidationError):
    """Raised when the response is not a valid JSON object."""


class SchemaValidationError(ResponseValidationError):
    """Raised when valid JSON does not match the Pydantic schema."""


class UngroundedValueError(ResponseValidationError):
    """Raised when a value does not appear in the source document.

    This is how the pipeline detects fabricated information.
    """


class MaxRetriesExceededError(ExtractorError):
    """Raised when the LLM never produced a valid response within the limit."""

    def __init__(self, attempts: int, last_errors: Sequence[str]) -> None:
        self.attempts = attempts
        self.last_errors: list[str] = list(last_errors)
        details = "; ".join(self.last_errors[:5]) or "no details available"
        super().__init__(
            f"No valid response after {attempts} attempt(s). "
            f"Last problems: {details}"
        )
