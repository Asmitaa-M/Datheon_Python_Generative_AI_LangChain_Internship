"""Strict Pydantic schemas for invoices and resumes.

Design rules (these are what "strict" means in this project):

* Every key must be present. A value that is not in the document must be an
  explicit ``null`` (or ``[]`` for lists), never a missing key.
* Unexpected keys are rejected (``extra="forbid"``).
* Numbers must be real JSON numbers. The string ``"12.50"`` is rejected, and so
  are booleans, ``NaN`` and ``Infinity``.
* Empty strings are rejected. "Not stated" is ``null``.
* Dates on invoices must be real ISO dates (``YYYY-MM-DD``).
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Annotated, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
)

from exceptions import UnsupportedDocumentTypeError

SUPPORTED_DOCUMENT_TYPES = ("invoice", "resume")


# --------------------------------------------------------------------------- #
# Reusable validated types
# --------------------------------------------------------------------------- #
_EMAIL_DOMAIN = r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
_EMAIL_RE = re.compile(rf"^[A-Za-z0-9._%+\-]+@{_EMAIL_DOMAIN}$")
_PHONE_RE = re.compile(r"^\+?[\d(][\d\s().\-]*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_email(value: str) -> str:
    local_part = value.split("@", 1)[0]
    if (
        not _EMAIL_RE.match(value)
        or ".." in local_part
        or local_part.startswith(".")
        or local_part.endswith(".")
    ):
        raise ValueError("is not a valid email address")
    return value


def _validate_phone(value: str) -> str:
    digit_count = len(re.sub(r"\D", "", value))
    if not _PHONE_RE.match(value) or not 7 <= digit_count <= 15:
        raise ValueError(
            "is not a valid phone number (expected 7-15 digits, "
            "optionally with +, spaces, dashes or parentheses)"
        )
    return value


def _validate_iso_date(value: str) -> str:
    if not _ISO_DATE_RE.match(value):
        raise ValueError("must be an ISO date in YYYY-MM-DD format")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError("is not a real calendar date") from None
    return value


_STRIPPED = StringConstraints(strict=True, strip_whitespace=True)

NonEmptyStr = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]
EmailStr = Annotated[str, _STRIPPED, AfterValidator(_validate_email)]
PhoneStr = Annotated[str, _STRIPPED, AfterValidator(_validate_phone)]
IsoDateStr = Annotated[str, _STRIPPED, AfterValidator(_validate_iso_date)]
CurrencyCode = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, pattern=r"^[A-Z]{3}$")
]

# strict=True: JSON numbers only (ints are accepted as floats; strings, bools,
# NaN and Infinity are not).
NonNegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
# Line-item amounts may legitimately be negative (discounts, credits).
SignedNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class StrictModel(BaseModel):
    """Base model: unknown keys are rejected and defaults are validated."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


# --------------------------------------------------------------------------- #
# Invoice
# --------------------------------------------------------------------------- #
class PaymentStatus(str, Enum):
    """Payment states an invoice can explicitly declare."""

    PAID = "paid"
    UNPAID = "unpaid"
    PARTIALLY_PAID = "partially_paid"
    OVERDUE = "overdue"


class LineItem(StrictModel):
    """One row of an invoice."""

    description: NonEmptyStr
    quantity: Optional[PositiveNumber]
    unit_price: Optional[NonNegativeNumber]
    amount: Optional[SignedNumber]

    @field_serializer("quantity")
    def _serialize_quantity(self, value: Optional[float]) -> Union[int, float, None]:
        """Write whole quantities as ``3`` instead of ``3.0`` in the JSON."""
        if value is not None and float(value).is_integer():
            return int(value)
        return value


class InvoiceData(StrictModel):
    """Structured data extracted from an invoice."""

    invoice_number: Optional[NonEmptyStr]
    invoice_date: Optional[IsoDateStr]
    due_date: Optional[IsoDateStr]
    vendor_name: Optional[NonEmptyStr]
    vendor_address: Optional[NonEmptyStr]
    customer_name: Optional[NonEmptyStr]
    customer_address: Optional[NonEmptyStr]
    subtotal: Optional[NonNegativeNumber]
    tax: Optional[NonNegativeNumber]
    total_amount: Optional[NonNegativeNumber]
    currency: Optional[CurrencyCode]
    payment_status: Optional[PaymentStatus]
    line_items: list[LineItem]


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
class Education(StrictModel):
    """One education entry. Dates are kept as written (e.g. ``2019``)."""

    institution: NonEmptyStr
    degree: Optional[NonEmptyStr]
    field_of_study: Optional[NonEmptyStr]
    start_date: Optional[NonEmptyStr]
    end_date: Optional[NonEmptyStr]


class WorkExperience(StrictModel):
    """One job. Dates are kept as written (e.g. ``Jan 2021``, ``Present``)."""

    job_title: NonEmptyStr
    company: NonEmptyStr
    location: Optional[NonEmptyStr]
    start_date: Optional[NonEmptyStr]
    end_date: Optional[NonEmptyStr]
    responsibilities: list[NonEmptyStr]


class Project(StrictModel):
    """One personal or professional project."""

    name: NonEmptyStr
    description: Optional[NonEmptyStr]
    technologies: list[NonEmptyStr]


class Certification(StrictModel):
    """One certification."""

    name: NonEmptyStr
    issuer: Optional[NonEmptyStr]
    date: Optional[NonEmptyStr]


class ResumeData(StrictModel):
    """Structured data extracted from a resume."""

    full_name: Optional[NonEmptyStr]
    email: Optional[EmailStr]
    phone: Optional[PhoneStr]
    location: Optional[NonEmptyStr]
    summary: Optional[NonEmptyStr]
    skills: list[NonEmptyStr]
    education: list[Education]
    work_experience: list[WorkExperience]
    projects: list[Project]
    certifications: list[Certification]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceData,
    "resume": ResumeData,
}


def resolve_document_type(name: object) -> str:
    """Normalise a user-supplied document type (``" Invoice "`` -> ``"invoice"``).

    Raises:
        UnsupportedDocumentTypeError: If the type is not invoice or resume.
    """
    normalized = name.strip().lower() if isinstance(name, str) else ""
    if normalized not in _SCHEMAS:
        supported = ", ".join(SUPPORTED_DOCUMENT_TYPES)
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type {name!r}. Supported types: {supported}."
        )
    return normalized


def get_schema(document_type: str) -> type[BaseModel]:
    """Return the Pydantic model for a (valid) document type."""
    return _SCHEMAS[resolve_document_type(document_type)]
