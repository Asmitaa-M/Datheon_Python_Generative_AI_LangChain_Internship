"""Tests for JSON parsing, schema validation and fabrication (grounding) checks."""

from __future__ import annotations

import json

import pytest

from exceptions import (
    InvalidJSONError,
    ResponseValidationError,
    SchemaValidationError,
    UngroundedValueError,
)
from schemas import InvoiceData, ResumeData
from validator import (
    find_ungrounded_values,
    parse_json_response,
    strip_code_fences,
    validate_against_schema,
    validate_response,
)


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #
def test_parse_valid_json_object():
    assert parse_json_response('{"a": 1, "b": [true, null]}') == {"a": 1, "b": [True, None]}


def test_parse_tolerates_surrounding_whitespace():
    assert parse_json_response('\n\n  {"a": 1}  \n') == {"a": 1}


@pytest.mark.parametrize(
    "wrapped",
    ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '  ```JSON\n{"a": 1}```  '],
)
def test_parse_removes_accidental_code_fences(wrapped):
    assert parse_json_response(wrapped) == {"a": 1}


def test_strip_code_fences_leaves_plain_text_alone():
    assert strip_code_fences('  {"a": 1} ') == '{"a": 1}'


@pytest.mark.parametrize(
    "malformed",
    [
        "{",
        '{"a": 1,}',
        "{'a': 1}",
        '{"a": }',
        'Here is the JSON: {"a": 1}',
        '{"a": 1} trailing words',
        "not json at all",
        '{"a": 1}{"b": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
        '{"a": "unterminated}',
    ],
)
def test_malformed_json_raises_invalid_json_error(malformed):
    with pytest.raises(InvalidJSONError) as caught:
        parse_json_response(malformed)
    assert caught.value.errors
    assert "not valid JSON" in caught.value.errors[0]


def test_malformed_json_error_reports_position():
    with pytest.raises(InvalidJSONError) as caught:
        parse_json_response('{\n  "a": 1,\n  "b": \n}')
    assert "line 4" in caught.value.errors[0]


@pytest.mark.parametrize("empty", ["", "   ", "\n\t", None])
def test_empty_response_is_rejected(empty):
    with pytest.raises(InvalidJSONError) as caught:
        parse_json_response(empty)
    assert "empty" in caught.value.errors[0].lower()


@pytest.mark.parametrize("not_an_object", ["[1, 2]", "42", '"text"', "null", "true"])
def test_top_level_value_must_be_an_object(not_an_object):
    with pytest.raises(InvalidJSONError) as caught:
        parse_json_response(not_an_object)
    assert "object" in caught.value.errors[0]


# --------------------------------------------------------------------------- #
# Schema validation and error messages
# --------------------------------------------------------------------------- #
def test_validate_against_schema_returns_model(invoice_payload):
    model = validate_against_schema(invoice_payload, InvoiceData)
    assert isinstance(model, InvoiceData)


def test_error_messages_classify_each_kind_of_problem(invoice_payload):
    invoice_payload["total_amount"] = "12.50"  # wrong type
    invoice_payload["currency"] = "usd"  # invalid value
    invoice_payload["surprise"] = 1  # unexpected field
    del invoice_payload["tax"]  # missing field
    invoice_payload["line_items"][0]["quantity"] = -2  # nested invalid value

    with pytest.raises(SchemaValidationError) as caught:
        validate_against_schema(invoice_payload, InvoiceData)

    by_field = {line.split(":")[0]: line for line in caught.value.errors}
    assert "invalid type" in by_field["total_amount"]
    assert "'12.50'" in by_field["total_amount"]  # shows what was received
    assert "invalid value" in by_field["currency"]
    assert "unexpected field" in by_field["surprise"]
    assert "missing required field" in by_field["tax"]
    assert "null" in by_field["tax"]  # tells the model how to fix it
    assert "invalid value" in by_field["line_items.0.quantity"]
    assert len(caught.value.errors) == 5


def test_long_invalid_values_are_shortened_in_messages(resume_payload):
    resume_payload["email"] = "x" * 500
    with pytest.raises(SchemaValidationError) as caught:
        validate_against_schema(resume_payload, ResumeData)
    assert len(caught.value.errors[0]) < 300


# --------------------------------------------------------------------------- #
# validate_response (all layers together)
# --------------------------------------------------------------------------- #
def test_validate_response_success_for_invoice(invoice_payload, invoice_text):
    model = validate_response(json.dumps(invoice_payload), "invoice", invoice_text)
    assert model.total_amount == 77290.59


def test_validate_response_success_for_resume(resume_payload, resume_text):
    model = validate_response(json.dumps(resume_payload), "resume", resume_text)
    assert model.full_name == "Priya Ramanathan"


def test_validate_response_without_source_skips_grounding(invoice_payload):
    invoice_payload["invoice_number"] = "NOT-IN-ANY-DOCUMENT"
    assert validate_response(json.dumps(invoice_payload), "invoice").invoice_number


@pytest.mark.parametrize(
    "raw,expected_error",
    [
        ("{", InvalidJSONError),
        ("[]", InvalidJSONError),
        ('{"invoice_number": "x"}', SchemaValidationError),
    ],
)
def test_validate_response_raises_specific_errors(raw, expected_error, invoice_text):
    with pytest.raises(expected_error) as caught:
        validate_response(raw, "invoice", invoice_text)
    assert isinstance(caught.value, ResponseValidationError)


# --------------------------------------------------------------------------- #
# Grounding: prevention of fabricated information
# --------------------------------------------------------------------------- #
def test_fabricated_email_is_detected(resume_payload, resume_text):
    resume_payload["email"] = "invented.person@fake-domain.org"
    with pytest.raises(UngroundedValueError) as caught:
        validate_response(json.dumps(resume_payload), "resume", resume_text)
    assert caught.value.errors[0].startswith("email:")
    assert "use null" in caught.value.errors[0]


def test_fabricated_phone_is_detected(resume_payload, resume_text):
    resume_payload["phone"] = "+1 202 555 0199"
    with pytest.raises(UngroundedValueError) as caught:
        validate_response(json.dumps(resume_payload), "resume", resume_text)
    assert caught.value.errors[0].startswith("phone:")


def test_fabricated_invoice_number_and_vendor_are_detected(invoice_payload, invoice_text):
    invoice_payload["invoice_number"] = "INV-0000-FAKE"
    invoice_payload["vendor_name"] = "Imaginary Industries"
    with pytest.raises(UngroundedValueError) as caught:
        validate_response(json.dumps(invoice_payload), "invoice", invoice_text)
    fields = {line.split(":")[0] for line in caught.value.errors}
    assert fields == {"invoice_number", "vendor_name"}


def test_calculated_total_not_in_document_is_detected(invoice_payload):
    """A model that adds numbers up instead of reading them is making an assumption."""
    source = "Invoice INV-2026-0042\nVendor: Brightline Software Solutions\nSubtotal: 100.00\nTax: 18.00\n"
    invoice_payload["subtotal"] = 100.0
    invoice_payload["tax"] = 18.0
    invoice_payload["total_amount"] = 118.0  # never printed in the document
    invoice_payload["customer_name"] = None
    invoice_payload["line_items"] = []
    with pytest.raises(UngroundedValueError) as caught:
        validate_response(json.dumps(invoice_payload), "invoice", source)
    assert [line.split(":")[0] for line in caught.value.errors] == ["total_amount"]


def test_fabricated_line_item_numbers_are_detected(invoice_payload, invoice_text):
    invoice_payload["line_items"][0]["unit_price"] = 123456.78
    with pytest.raises(UngroundedValueError) as caught:
        validate_response(json.dumps(invoice_payload), "invoice", invoice_text)
    assert caught.value.errors[0].startswith("line_items.0.unit_price:")


def test_null_values_are_never_flagged(invoice_payload):
    for name in InvoiceData.model_fields:
        invoice_payload[name] = None
    invoice_payload["line_items"] = []
    assert validate_response(json.dumps(invoice_payload), "invoice", "any text at all")


def test_amounts_match_regardless_of_thousands_separator_style(invoice_payload):
    invoice_payload.update(subtotal=1234567.5, tax=None, total_amount=None, line_items=[])
    invoice_payload.update(invoice_number=None, vendor_name=None, customer_name=None)
    for printed in ("1,234,567.50", "12,34,567.50", "1.234.567,50", "1234567.50"):
        model = validate_response(
            json.dumps(invoice_payload), "invoice", f"Subtotal: {printed}"
        )
        assert model.subtotal == 1234567.5


def test_negative_discount_matches_printed_amount(invoice_payload):
    invoice_payload.update(subtotal=None, tax=None, total_amount=None)
    invoice_payload.update(invoice_number=None, vendor_name=None, customer_name=None)
    invoice_payload["line_items"] = [
        {"description": "Loyalty discount", "quantity": 1, "unit_price": None, "amount": -500.0}
    ]
    assert validate_response(json.dumps(invoice_payload), "invoice", "Loyalty discount 1  -500.00")


def test_names_match_ignoring_case_and_extra_whitespace(resume_payload):
    resume_payload["full_name"] = "Priya   Ramanathan"
    source = "PRIYA RAMANATHAN\nEmail: priya.ramanathan@example.com\nPhone: +91 98765 43210"
    assert validate_response(json.dumps(resume_payload), "resume", source).full_name


def test_unicode_names_are_grounded(resume_payload):
    resume_payload["full_name"] = "Anaïs Müller"
    source = "ANAÏS MÜLLER\nanaïs@example.com\nEmail: priya.ramanathan@example.com\nPhone: 98765 43210"
    resume_payload["phone"] = "98765 43210"
    assert validate_response(json.dumps(resume_payload), "resume", source).full_name == "Anaïs Müller"


def test_phone_matches_despite_different_formatting(resume_payload):
    resume_payload["phone"] = "+919876543210"
    source = "Priya Ramanathan priya.ramanathan@example.com\nCall +91 98765-43210"
    assert validate_response(json.dumps(resume_payload), "resume", source).phone


def test_find_ungrounded_values_returns_empty_list_when_clean(invoice_payload, invoice_text):
    model = InvoiceData.model_validate(invoice_payload)
    assert find_ungrounded_values(model, "invoice", invoice_text) == []


def test_every_ungrounded_value_is_listed_in_one_pass(invoice_payload):
    model = InvoiceData.model_validate(invoice_payload)
    problems = find_ungrounded_values(model, "invoice", "completely unrelated text 7 8 9")
    fields = {line.split(":")[0] for line in problems}
    assert {"invoice_number", "vendor_name", "customer_name", "subtotal", "tax", "total_amount"} <= fields
