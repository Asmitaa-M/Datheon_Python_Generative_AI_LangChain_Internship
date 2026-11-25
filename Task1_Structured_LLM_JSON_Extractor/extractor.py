"""The extraction pipeline: prompt -> LLM -> validate -> (correct) -> result.

Two independent retry budgets keep the loop safe and bounded:

* ``max_retries``     - correction attempts when the response is malformed or
  invalid. Total LLM answers considered: ``1 + max_retries``.
* ``api_max_retries`` - retries (with exponential backoff) for temporary API
  problems such as rate limits or network errors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from exceptions import (
    LLMTransientError,
    MaxRetriesExceededError,
    ResponseValidationError,
    SchemaValidationError,
)
from llm_client import BaseLLMClient
from prompt_templates import build_correction_prompt, build_extraction_prompt
from schemas import get_schema, resolve_document_type
from utils import PathLike, ensure_non_empty_text, write_json_file
from validator import format_validation_errors, validate_response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """A successfully validated extraction."""

    document_type: str
    data: BaseModel
    attempts: int  # number of LLM answers that were needed (1 = first try)

    def to_dict(self) -> dict[str, Any]:
        """Return the data as plain JSON-compatible Python objects."""
        return self.data.model_dump(mode="json")


class DocumentExtractor:
    """Extract structured data from unstructured text using an LLM client."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_retries: int = 3,
        api_max_retries: int = 2,
        api_retry_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0 or api_max_retries < 0:
            raise ValueError("Retry limits must be zero or greater.")
        self._llm_client = llm_client
        self._max_retries = max_retries
        self._api_max_retries = api_max_retries
        self._api_retry_delay = api_retry_delay
        self._sleep = sleep

    def extract(self, document_type: str, text: str) -> ExtractionResult:
        """Extract and validate structured data from ``text``.

        Args:
            document_type: ``"invoice"`` or ``"resume"``.
            text: The unstructured document text.

        Raises:
            UnsupportedDocumentTypeError: Unknown document type.
            EmptyInputError: The text is empty or whitespace only.
            LLMError: The API failed and could not be recovered by retrying.
            MaxRetriesExceededError: No valid response within the retry limit.
        """
        resolved_type = resolve_document_type(document_type)
        document_text = ensure_non_empty_text(text)

        base_prompt = build_extraction_prompt(resolved_type, document_text)
        prompt = base_prompt
        last_errors: list[str] = []
        total_attempts = self._max_retries + 1

        for attempt in range(1, total_attempts + 1):
            logger.info("Extraction attempt %d of %d.", attempt, total_attempts)
            raw_response = self._call_llm(prompt)
            try:
                model = validate_response(
                    raw_response, resolved_type, source_text=document_text
                )
            except ResponseValidationError as exc:
                last_errors = exc.errors
                logger.warning(
                    "Attempt %d rejected (%s): %d problem(s).",
                    attempt,
                    type(exc).__name__,
                    len(exc.errors),
                )
                # Always rebuild from the ORIGINAL prompt so the source
                # document is preserved and prompts do not keep growing.
                prompt = build_correction_prompt(base_prompt, raw_response, exc.errors)
                continue

            logger.info("Valid response received on attempt %d.", attempt)
            return ExtractionResult(
                document_type=resolved_type, data=model, attempts=attempt
            )

        raise MaxRetriesExceededError(total_attempts, last_errors)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM, retrying temporary API failures with backoff."""
        api_attempt = 0
        while True:
            try:
                return self._llm_client.generate(prompt)
            except LLMTransientError as exc:
                if api_attempt >= self._api_max_retries:
                    raise
                delay = self._api_retry_delay * (2**api_attempt)
                api_attempt += 1
                logger.warning(
                    "Temporary API problem (%s). Retry %d of %d in %.1fs.",
                    type(exc).__name__,
                    api_attempt,
                    self._api_max_retries,
                    delay,
                )
                self._sleep(delay)


def save_result(result: ExtractionResult, path: PathLike) -> Path:
    """Validate a result once more and save it as UTF-8 JSON.

    Raises:
        SchemaValidationError: If the data no longer matches its schema. Nothing
            is written in that case.
        OutputFileError: If the file cannot be written.
    """
    payload = result.to_dict()
    try:
        get_schema(result.document_type).model_validate(payload)
    except ValidationError as exc:
        problems = format_validation_errors(exc)
        raise SchemaValidationError(
            "Refusing to save: result does not match its schema.", problems
        ) from exc
    return write_json_file(payload, path)
