"""Tests for the strict Pydantic schemas."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from pydantic import ValidationError

from exceptions import UnsupportedDocumentTypeError
from schemas import (
    InvoiceData,
    LineItem,
    PaymentStatus,
    ResumeData,
    get_schema,
    resolve_document_type,
)


def error_types(error: ValidationError) -> set[str]:
    return {item["type"] for item in error.errors()}


def error_locations(error: ValidationError) -> set[str]:
    return {".".join(str(part) for part in item["loc"]) for item in error.errors()}


# --------------------------------------------------------------------------- #
# Valid data
# --------------------------------------------------------------------------- #
def test_valid_invoice_is_accepted(invoice_payload):
    invoice = InvoiceData.model_validate(invoice_payload)
    assert invoice.invoice_number == invoice_payload["invoice_number"]
    assert invoice.payment_status is PaymentStatus.UNPAID
    assert len(invoice.line_items) == len(invoice_payload["line_items"])
    assert invoice.line_items[1].unit_price == invoice_payload["line_items"][1]["unit_price"]


def test_valid_resume_is_accepted(resume_payload):
    resume = ResumeData.model_validate(resume_payload)
    assert resume.email == resume_payload["email"]
    assert resume.work_experience[0].end_date == "Present"
    assert resume.projects[0].technologies == ["Python", "Flask"]


def test_invoice_round_trips_through_json_dump(invoice_payload):
    invoice = InvoiceData.model_validate(invoice_payload)
    dumped = invoice.model_dump(mode="json")
    assert InvoiceData.model_validate(dumped) == invoice
    assert dumped["payment_status"] == "unpaid"  # plain string, not an Enum


def test_resume_round_trips_through_json_dump(resume_payload):
    resume = ResumeData.model_validate(resume_payload)
    assert ResumeData.model_validate(resume.model_dump(mode="json")) == resume


def test_whole_quantities_are_written_without_decimal_point(invoice_payload):
    dumped = InvoiceData.model_validate(invoice_payload).model_dump(mode="json")
    assert isinstance(dumped["line_items"][0]["quantity"], int)


def test_fractional_quantity_is_preserved(invoice_payload):
    invoice_payload["line_items"][0]["quantity"] = 1.5
    dumped = InvoiceData.model_validate(invoice_payload).model_dump(mode="json")
    assert dumped["line_items"][0]["quantity"] == 1.5


def test_unicode_text_is_accepted(invoice_payload, resume_payload):
    invoice_payload["vendor_name"] = "Müller & Söhne GmbH €"
    resume_payload["full_name"] = "Anaïs Ñandú 田中"
    assert InvoiceData.model_validate(invoice_payload).vendor_name.startswith("Müller")
    assert ResumeData.model_validate(resume_payload).full_name.endswith("田中")


def test_whitespace_around_strings_is_trimmed(invoice_payload):
    invoice_payload["vendor_name"] = "  Padded Name  "
    assert InvoiceData.model_validate(invoice_payload).vendor_name == "Padded Name"


# --------------------------------------------------------------------------- #
# Missing optional fields (null) and missing keys
# --------------------------------------------------------------------------- #
def test_invoice_with_all_optional_fields_null_is_valid():
    payload: dict[str, Any] = {name: None for name in InvoiceData.model_fields}
    payload["line_items"] = []
    invoice = InvoiceData.model_validate(payload)
    assert invoice.total_amount is None
    assert invoice.line_items == []


def test_resume_with_all_optional_fields_null_is_valid():
    payload: dict[str, Any] = {name: None for name in ResumeData.model_fields}
    for list_field in (
        "skills",
        "education",
        "work_experience",
        "projects",
        "certifications",
    ):
        payload[list_field] = []
    resume = ResumeData.model_validate(payload)
    assert resume.email is None and resume.skills == []


def test_line_item_amounts_may_be_null():
    item = LineItem.model_validate(
        {"description": "Consulting", "quantity": None, "unit_price": None, "amount": None}
    )
    assert item.amount is None


@pytest.mark.parametrize("missing_key", list(InvoiceData.model_fields))
def test_invoice_missing_key_is_rejected(invoice_payload, missing_key):
    del invoice_payload[missing_key]
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_types(caught.value) == {"missing"}
    assert error_locations(caught.value) == {missing_key}


@pytest.mark.parametrize("missing_key", list(ResumeData.model_fields))
def test_resume_missing_key_is_rejected(resume_payload, missing_key):
    del resume_payload[missing_key]
    with pytest.raises(ValidationError) as caught:
        ResumeData.model_validate(resume_payload)
    assert error_types(caught.value) == {"missing"}


def test_line_item_missing_key_is_rejected(invoice_payload):
    del invoice_payload["line_items"][0]["amount"]
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {"line_items.0.amount"}


# --------------------------------------------------------------------------- #
# Invalid email addresses and phone numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "email",
    [
        "not-an-email",
        "missing-at.example.com",
        "two@@example.com",
        "no-domain@",
        "@no-local.com",
        "spaces in@example.com",
        "no-tld@example",
        "dots..double@example.com",
        ".leading@example.com",
        "trailing.@example.com",
        "a@b.c",
        "",
        "   ",
        12345,
        ["a@example.com"],
    ],
)
def test_invalid_emails_are_rejected(resume_payload, email):
    resume_payload["email"] = email
    with pytest.raises(ValidationError) as caught:
        ResumeData.model_validate(resume_payload)
    assert error_locations(caught.value) == {"email"}


@pytest.mark.parametrize(
    "email",
    ["a@example.com", "first.last+tag@sub.example.co.uk", "USER_1@Example.ORG", "x-y@a-b.io"],
)
def test_valid_emails_are_accepted(resume_payload, email):
    resume_payload["email"] = email
    assert ResumeData.model_validate(resume_payload).email == email


@pytest.mark.parametrize(
    "phone",
    ["abc", "12345", "+", "call me", "1234567890123456", "12-ab-34567", "", 5551234567],
)
def test_invalid_phones_are_rejected(resume_payload, phone):
    resume_payload["phone"] = phone
    with pytest.raises(ValidationError) as caught:
        ResumeData.model_validate(resume_payload)
    assert error_locations(caught.value) == {"phone"}


@pytest.mark.parametrize(
    "phone", ["+91 98765 43210", "(555) 123-4567", "555-123-4567", "+44 7700 900123"]
)
def test_valid_phones_are_accepted(resume_payload, phone):
    resume_payload["phone"] = phone
    assert ResumeData.model_validate(resume_payload).phone == phone


# --------------------------------------------------------------------------- #
# Invalid numerical values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["subtotal", "tax", "total_amount"])
@pytest.mark.parametrize(
    "bad_value",
    ["12.50", "", True, False, -0.01, -100, math.nan, math.inf, -math.inf, [1], {"v": 1}],
)
def test_invoice_rejects_invalid_amounts(invoice_payload, field, bad_value):
    invoice_payload[field] = bad_value
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {field}


@pytest.mark.parametrize("zero_or_integer", [0, 0.0, 1, 250])
def test_invoice_accepts_zero_and_integer_amounts(invoice_payload, zero_or_integer):
    invoice_payload["total_amount"] = zero_or_integer
    assert InvoiceData.model_validate(invoice_payload).total_amount == zero_or_integer


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("quantity", 0),
        ("quantity", -1),
        ("quantity", "2"),
        ("quantity", True),
        ("unit_price", -5),
        ("unit_price", "9.99"),
        ("amount", "100"),
        ("amount", math.nan),
    ],
)
def test_line_item_rejects_invalid_numbers(invoice_payload, field, bad_value):
    invoice_payload["line_items"][0][field] = bad_value
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {f"line_items.0.{field}"}


def test_negative_line_item_amount_is_allowed_for_discounts(invoice_payload):
    invoice_payload["line_items"][0]["amount"] = -500.0
    assert InvoiceData.model_validate(invoice_payload).line_items[0].amount == -500.0


# --------------------------------------------------------------------------- #
# Dates, currency, payment status, empty strings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_date",
    [
        "2026-13-45",
        "2026-02-30",
        "05/10/2026",
        "2026-2-5",
        "20261005",
        "October 5, 2026",
        "2026-10-05T10:00:00",
        "",
        20261005,
    ],
)
@pytest.mark.parametrize("field", ["invoice_date", "due_date"])
def test_invoice_rejects_bad_dates(invoice_payload, field, bad_date):
    invoice_payload[field] = bad_date
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {field}


def test_leap_day_is_a_real_date(invoice_payload):
    invoice_payload["invoice_date"] = "2028-02-29"
    assert InvoiceData.model_validate(invoice_payload).invoice_date == "2028-02-29"


@pytest.mark.parametrize("bad_currency", ["usd", "US", "USDX", "$", "", 123, "U S"])
def test_invoice_rejects_bad_currency(invoice_payload, bad_currency):
    invoice_payload["currency"] = bad_currency
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {"currency"}


@pytest.mark.parametrize("bad_status", ["pending", "PAID", "Paid", "", 1, True])
def test_invoice_rejects_unknown_payment_status(invoice_payload, bad_status):
    invoice_payload["payment_status"] = bad_status
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {"payment_status"}


@pytest.mark.parametrize("status", ["paid", "unpaid", "partially_paid", "overdue"])
def test_invoice_accepts_every_documented_status(invoice_payload, status):
    invoice_payload["payment_status"] = status
    assert InvoiceData.model_validate(invoice_payload).payment_status.value == status


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_empty_strings_are_rejected_instead_of_null(invoice_payload, blank):
    invoice_payload["vendor_name"] = blank
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {"vendor_name"}


def test_empty_skill_string_is_rejected(resume_payload):
    resume_payload["skills"] = ["Python", ""]
    with pytest.raises(ValidationError):
        ResumeData.model_validate(resume_payload)


# --------------------------------------------------------------------------- #
# Unexpected fields and incorrect data types
# --------------------------------------------------------------------------- #
def test_invoice_rejects_unexpected_top_level_field(invoice_payload):
    invoice_payload["discount_code"] = "SAVE10"
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_types(caught.value) == {"extra_forbidden"}
    assert error_locations(caught.value) == {"discount_code"}


def test_invoice_rejects_unexpected_line_item_field(invoice_payload):
    invoice_payload["line_items"][0]["sku"] = "X-1"
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert error_locations(caught.value) == {"line_items.0.sku"}


def test_resume_rejects_unexpected_nested_field(resume_payload):
    resume_payload["education"][0]["gpa"] = 3.9
    with pytest.raises(ValidationError) as caught:
        ResumeData.model_validate(resume_payload)
    assert error_locations(caught.value) == {"education.0.gpa"}


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("invoice_number", 12345),
        ("vendor_name", ["Acme"]),
        ("line_items", {"description": "x"}),
        ("line_items", "none"),
        ("line_items", None),
        ("subtotal", "65,500.50"),
    ],
)
def test_invoice_rejects_incorrect_types(invoice_payload, field, bad_value):
    invoice_payload[field] = bad_value
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(invoice_payload)
    assert field in {loc.split(".")[0] for loc in error_locations(caught.value)}


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("skills", "Python, SQL"),
        ("skills", None),
        ("skills", [1, 2]),
        ("education", "Anna University"),
        ("work_experience", {"company": "x"}),
        ("full_name", 42),
        ("certifications", ["Python for Data Science"]),
    ],
)
def test_resume_rejects_incorrect_types(resume_payload, field, bad_value):
    resume_payload[field] = bad_value
    with pytest.raises(ValidationError) as caught:
        ResumeData.model_validate(resume_payload)
    assert field in {loc.split(".")[0] for loc in error_locations(caught.value)}


def test_non_object_payload_is_rejected():
    with pytest.raises(ValidationError):
        InvoiceData.model_validate(["not", "an", "object"])


def test_every_problem_is_reported_not_just_the_first(invoice_payload):
    broken = copy.deepcopy(invoice_payload)
    broken["total_amount"] = "oops"
    broken["currency"] = "usd"
    broken["surprise"] = 1
    del broken["tax"]
    with pytest.raises(ValidationError) as caught:
        InvoiceData.model_validate(broken)
    assert error_locations(caught.value) == {"total_amount", "currency", "surprise", "tax"}


# --------------------------------------------------------------------------- #
# Document type registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "given,expected",
    [("invoice", "invoice"), ("  Invoice ", "invoice"), ("RESUME", "resume"), ("resume\n", "resume")],
)
def test_document_type_is_normalised(given, expected):
    assert resolve_document_type(given) == expected


@pytest.mark.parametrize("bad", ["contract", "", "   ", "invoices", "cv", None, 7, ["invoice"]])
def test_unsupported_document_type_is_rejected(bad):
    with pytest.raises(UnsupportedDocumentTypeError) as caught:
        resolve_document_type(bad)
    assert "invoice" in str(caught.value) and "resume" in str(caught.value)


def test_get_schema_returns_matching_models():
    assert get_schema("invoice") is InvoiceData
    assert get_schema(" Resume") is ResumeData
    with pytest.raises(UnsupportedDocumentTypeError):
        get_schema("receipt")
