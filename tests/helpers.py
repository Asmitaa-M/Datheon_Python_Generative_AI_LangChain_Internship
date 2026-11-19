"""Shared test helpers: a scripted fake LLM and document text generators."""

from __future__ import annotations

import json
from typing import Any, Sequence, Union

from llm_client import BaseLLMClient

Scripted = Union[str, dict, list, BaseException]


class ScriptedLLM(BaseLLMClient):
    """Fake LLM that replays a fixed list of responses and records prompts.

    Each scripted item is either raw text, a dict/list (sent as JSON), or an
    exception instance (raised). Asking for more responses than were scripted
    fails the test, which also proves that retry loops are bounded.
    """

    def __init__(self, *responses: Scripted) -> None:
        self._responses: list[Scripted] = list(responses)
        self.prompts: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLM was called more often than scripted")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, (dict, list)):
            return json.dumps(item)
        return item


def money(value: float) -> str:
    """Format like a printed invoice: 1,234.50."""
    return f"{value:,.2f}"


def make_invoice_text(
    number: str = "INV-9",
    invoice_date: str = "2025-01-31",
    due_date: Union[str, None] = "2025-02-28",
    vendor: str = "Acme Ltd",
    customer: str = "Zed Corp",
    items: Sequence[tuple[str, int, float]] = (("Widget", 2, 10.5),),
    tax: Union[float, None] = 3.15,
    currency: Union[str, None] = "USD",
    status: Union[str, None] = "Paid",
) -> str:
    """Render an invoice in the layout the mock parser understands.

    The values come from the arguments, so tests can compare the extraction
    against the arguments instead of against a hardcoded expected answer.
    """
    subtotal = sum(quantity * price for _, quantity, price in items)
    lines = ["INVOICE", "", f"Invoice Number: {number}", f"Invoice Date: {invoice_date}"]
    if due_date:
        lines.append(f"Due Date: {due_date}")
    lines += ["", f"Vendor: {vendor}", f"Bill To: {customer}", ""]
    lines.append("Description | Qty | Unit Price | Amount")
    for description, quantity, price in items:
        lines.append(
            f"{description} | {quantity} | {money(price)} | {money(quantity * price)}"
        )
    lines += ["", f"Subtotal: {money(subtotal)}"]
    if tax is not None:
        lines.append(f"Tax (VAT): {money(tax)}")
        lines.append(f"Total Amount: {money(subtotal + tax)}")
    else:
        lines.append(f"Total Amount: {money(subtotal)}")
    if currency:
        lines.append(f"Currency: {currency}")
    if status:
        lines.append(f"Payment Status: {status}")
    return "\n".join(lines) + "\n"


def make_resume_text(
    name: str = "Alex Doe",
    email: Union[str, None] = "alex.doe@example.com",
    phone: Union[str, None] = "+1 415 555 0100",
    location: Union[str, None] = "Austin, USA",
    skills: Sequence[str] = ("Python", "SQL"),
    summary: Union[str, None] = "Engineer who likes tests.",
) -> str:
    """Render a small resume in the layout the mock parser understands."""
    contact = []
    if email:
        contact.append(f"Email: {email}")
    if phone:
        contact.append(f"Phone: {phone}")
    lines: list[Any] = [name]
    if contact:
        lines.append(" | ".join(contact))
    if location:
        lines.append(f"Location: {location}")
    if summary:
        lines += ["", "SUMMARY", summary]
    if skills:
        lines += ["", "SKILLS", ", ".join(skills)]
    return "\n".join(lines) + "\n"
