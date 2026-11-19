"""Parsing and validating raw LLM output.

The pipeline has three layers, each producing clear error messages that can be
sent straight back to the LLM for a correction attempt:

1. :func:`parse_json_response`   - is it a JSON object at all?
2. :func:`validate_against_schema` - does it match the Pydantic schema?
3. :func:`find_ungrounded_values`  - are the identifying values and numbers
   really present in the source document? (Catches fabricated data.)
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from exceptions import (
    InvalidJSONError,
    SchemaValidationError,
    UngroundedValueError,
)
from schemas import InvoiceData, ResumeData, get_schema, resolve_document_type

_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?[ \t]*```\s*$", re.DOTALL)
_MAX_ECHOED_INPUT = 60


# --------------------------------------------------------------------------- #
# 1. JSON parsing
# --------------------------------------------------------------------------- #
def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence, if the model added one anyway."""
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text.strip()


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in JSON")


def parse_json_response(raw: Optional[str]) -> dict[str, Any]:
    """Parse an LLM response into a JSON object.

    Raises:
        InvalidJSONError: If the response is empty, not valid JSON, or not a
            top-level JSON object.
    """
    if not isinstance(raw, str) or not raw.strip():
        message = "The response was empty. Return a single JSON object."
        raise InvalidJSONError(message, [message])

    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        message = (
            f"The response is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        )
        raise InvalidJSONError(message, [message]) from exc
    except ValueError as exc:  # NaN / Infinity rejected by _reject_constant
        message = f"The response is not valid JSON: {exc}."
        raise InvalidJSONError(message, [message]) from exc

    if not isinstance(data, dict):
        message = (
            "The top-level JSON value must be an object, "
            f"but got {type(data).__name__}."
        )
        raise InvalidJSONError(message, [message])
    return data


# --------------------------------------------------------------------------- #
# 2. Schema validation
# --------------------------------------------------------------------------- #
def _describe_input(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_ECHOED_INPUT:
        text = text[: _MAX_ECHOED_INPUT - 3] + "..."
    return text


def format_validation_errors(error: ValidationError) -> list[str]:
    """Turn a Pydantic error into one readable line per problem.

    Each line names the field path and classifies the problem as a missing
    field, an unexpected field, an invalid type, or an invalid value.
    """
    lines: list[str] = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"]) or "<root>"
        error_type = item["type"]
        message = item["msg"]
        if error_type == "missing":
            lines.append(
                f"{path}: missing required field. Include it, using null "
                "if the document does not state it."
            )
        elif error_type == "extra_forbidden":
            lines.append(
                f"{path}: unexpected field that is not part of the schema. "
                "Remove it."
            )
        elif error_type.endswith("_type") or error_type == "model_attributes_type":
            lines.append(
                f"{path}: invalid type - {message} "
                f"(got {_describe_input(item.get('input'))})."
            )
        else:
            lines.append(
                f"{path}: invalid value - {message} "
                f"(got {_describe_input(item.get('input'))})."
            )
    return lines


def validate_against_schema(
    data: dict[str, Any], schema: type[BaseModel]
) -> BaseModel:
    """Validate a parsed JSON object against ``schema``.

    Raises:
        SchemaValidationError: With every problem listed in ``.errors``.
    """
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        problems = format_validation_errors(exc)
        raise SchemaValidationError(
            f"Response failed schema validation with {len(problems)} problem(s).",
            problems,
        ) from exc


# --------------------------------------------------------------------------- #
# 3. Grounding (fabrication) checks
# --------------------------------------------------------------------------- #
_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")


def _normalize_text(text: str) -> str:
    """Case-fold, apply Unicode normalisation and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _numbers_in(text: str) -> list[float]:
    """Collect every number in ``text``, understanding 1,234.50 and 1.234,50."""
    numbers: list[float] = []
    for token in _NUMBER_TOKEN_RE.findall(text):
        token = token.rstrip(".,")
        candidates = (
            token.replace(",", ""),  # 1,234.50 and 1,23,456.50
            token.replace(".", "").replace(",", "."),  # 1.234,50
        )
        for candidate in candidates:
            try:
                numbers.append(float(candidate))
            except ValueError:
                continue
    return numbers


class _SourceIndex:
    """Pre-computed views of the source document used for grounding checks."""

    def __init__(self, source_text: str) -> None:
        self.text = _normalize_text(source_text)
        self.digits = re.sub(r"\D", "", source_text)
        self.numbers = _numbers_in(source_text)

    def has_text(self, value: str) -> bool:
        return _normalize_text(value) in self.text

    def has_digits(self, value: str) -> bool:
        digits = re.sub(r"\D", "", value)
        return bool(digits) and digits in self.digits

    def has_number(self, value: float) -> bool:
        target = abs(value)
        return any(
            math.isclose(target, found, rel_tol=1e-9, abs_tol=1e-6)
            for found in self.numbers
        )


def _ungrounded_message(path: str, value: Any) -> str:
    return (
        f"{path}: the value {_describe_input(value)} does not appear in the "
        "source document. Do not guess or calculate values; use null if the "
        "document does not state it."
    )


def _check_text(index: _SourceIndex, path: str, value: Optional[str]) -> list[str]:
    if value is None or index.has_text(value):
        return []
    return [_ungrounded_message(path, value)]


def _check_digits(index: _SourceIndex, path: str, value: Optional[str]) -> list[str]:
    if value is None or index.has_digits(value):
        return []
    return [_ungrounded_message(path, value)]


def _check_number(
    index: _SourceIndex, path: str, value: Optional[float]
) -> list[str]:
    if value is None or index.has_number(value):
        return []
    return [_ungrounded_message(path, value)]


def _ground_invoice(invoice: InvoiceData, index: _SourceIndex) -> list[str]:
    problems: list[str] = []
    for name in ("invoice_number", "vendor_name", "customer_name"):
        problems += _check_text(index, name, getattr(invoice, name))
    for name in ("subtotal", "tax", "total_amount"):
        problems += _check_number(index, name, getattr(invoice, name))
    for position, item in enumerate(invoice.line_items):
        for name in ("quantity", "unit_price", "amount"):
            problems += _check_number(
                index, f"line_items.{position}.{name}", getattr(item, name)
            )
    return problems


def _ground_resume(resume: ResumeData, index: _SourceIndex) -> list[str]:
    problems: list[str] = []
    problems += _check_text(index, "full_name", resume.full_name)
    problems += _check_text(index, "email", resume.email)
    problems += _check_digits(index, "phone", resume.phone)
    return problems


def find_ungrounded_values(
    model: BaseModel, document_type: str, source_text: str
) -> list[str]:
    """List identifying values and numbers that are NOT in the source document.

    Only objective values are checked (names, e-mail, phone, invoice number and
    every amount). Summaries and skills are left alone because a correct
    extraction may legitimately rephrase or tidy them.

    Returns:
        One message per problem; an empty list means everything is grounded.
    """
    resolved = resolve_document_type(document_type)
    index = _SourceIndex(source_text)
    if resolved == "invoice" and isinstance(model, InvoiceData):
        return _ground_invoice(model, index)
    if resolved == "resume" and isinstance(model, ResumeData):
        return _ground_resume(model, index)
    return []


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #
def validate_response(
    raw: Optional[str],
    document_type: str,
    source_text: Optional[str] = None,
) -> BaseModel:
    """Run all validation layers on a raw LLM response.

    Args:
        raw: The raw text returned by the LLM.
        document_type: ``"invoice"`` or ``"resume"``.
        source_text: The original document. When given, values that do not
            appear in it are rejected as fabricated.

    Raises:
        InvalidJSONError: The response is not a JSON object.
        SchemaValidationError: The JSON does not match the schema.
        UngroundedValueError: A value is not present in the source document.
    """
    data = parse_json_response(raw)
    model = validate_against_schema(data, get_schema(document_type))
    if source_text is not None:
        problems = find_ungrounded_values(model, document_type, source_text)
        if problems:
            raise UngroundedValueError(
                f"{len(problems)} value(s) do not appear in the source document.",
                problems,
            )
    return model
