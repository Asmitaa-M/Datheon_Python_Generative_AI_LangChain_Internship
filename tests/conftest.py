"""Fixtures shared by all tests.

Isolation guarantees given to every test automatically:

* Settings-related environment variables are removed, so a real API key on the
  developer's machine can never be picked up.
* All network connections are blocked, so a real (paid) API call is impossible.
* Logging configuration is restored after each test.
"""

from __future__ import annotations

import copy
import logging
import socket
from pathlib import Path
from typing import Any

import pytest

import utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

_SETTING_VARIABLES = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "LLM_MOCK_MODE",
    "MAX_RETRIES",
    "API_MAX_RETRIES",
    "API_RETRY_DELAY",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "OUTPUT_DIR",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Network access is blocked during tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    utils._active_handler = None


@pytest.fixture
def invoice_text() -> str:
    return (EXAMPLES_DIR / "sample_invoice.txt").read_text(encoding="utf-8")


@pytest.fixture
def resume_text() -> str:
    return (EXAMPLES_DIR / "sample_resume.txt").read_text(encoding="utf-8")


@pytest.fixture
def invoice_payload() -> dict[str, Any]:
    """A valid invoice answer matching examples/sample_invoice.txt."""
    return copy.deepcopy(
        {
            "invoice_number": "INV-2026-0042",
            "invoice_date": "2026-10-05",
            "due_date": "2026-11-04",
            "vendor_name": "Brightline Software Solutions",
            "vendor_address": "12 Residency Road, Bengaluru, Karnataka 560025, India",
            "customer_name": "Nilgiri Tea Traders",
            "customer_address": "48 Mount Road, Chennai, Tamil Nadu 600002, India",
            "subtotal": 65500.5,
            "tax": 11790.09,
            "total_amount": 77290.59,
            "currency": "INR",
            "payment_status": "unpaid",
            "line_items": [
                {
                    "description": "Website maintenance (monthly)",
                    "quantity": 3,
                    "unit_price": 15000.0,
                    "amount": 45000.0,
                },
                {
                    "description": "Cloud hosting – Standard plan",
                    "quantity": 1,
                    "unit_price": 8500.5,
                    "amount": 8500.5,
                },
            ],
        }
    )


@pytest.fixture
def resume_payload() -> dict[str, Any]:
    """A valid resume answer matching examples/sample_resume.txt."""
    return copy.deepcopy(
        {
            "full_name": "Priya Ramanathan",
            "email": "priya.ramanathan@example.com",
            "phone": "+91 98765 43210",
            "location": "Chennai, Tamil Nadu, India",
            "summary": "Backend developer with 3 years of experience.",
            "skills": ["Python", "SQL", "Docker"],
            "education": [
                {
                    "institution": "Anna University",
                    "degree": "B.E.",
                    "field_of_study": "Computer Science",
                    "start_date": "2018",
                    "end_date": "2022",
                }
            ],
            "work_experience": [
                {
                    "job_title": "Software Engineer",
                    "company": "Tamarind Technologies",
                    "location": "Chennai",
                    "start_date": "Jul 2022",
                    "end_date": "Present",
                    "responsibilities": ["Built REST APIs"],
                }
            ],
            "projects": [
                {
                    "name": "Expense Tracker",
                    "description": None,
                    "technologies": ["Python", "Flask"],
                }
            ],
            "certifications": [
                {"name": "Python for Data Science", "issuer": "Coursera", "date": "2023"}
            ],
        }
    )
