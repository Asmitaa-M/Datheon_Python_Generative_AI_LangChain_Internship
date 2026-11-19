"""Offline stand-in for the LLM, used by ``--mock`` / ``LLM_MOCK_MODE=true``.

This is NOT an AI model. It is a small rule-based parser that reads the
document out of the prompt and builds a JSON answer from it, so the whole
pipeline (prompt -> response -> validation -> saving) can be exercised without
an API key or any cost.

It understands the layout of the files in ``examples/`` (``Label: value``
lines, ``|``-separated tables and plain section headings). Anything it cannot
find becomes ``null`` / ``[]``, exactly like the prompts ask a real model to do.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from exceptions import LLMError
from llm_client import BaseLLMClient
from prompt_templates import DOCUMENT_END, DOCUMENT_START

_DOCUMENT_TYPE_RE = re.compile(r"^DOCUMENT_TYPE:\s*(\w+)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


# --------------------------------------------------------------------------- #
# Prompt parsing
# --------------------------------------------------------------------------- #
def parse_prompt(prompt: str) -> tuple[str, str]:
    """Return ``(document_type, document_text)`` found inside a prompt.

    Raises:
        LLMError: If the prompt does not follow the project's prompt layout.
    """
    type_match = _DOCUMENT_TYPE_RE.search(prompt)
    start = prompt.rfind(DOCUMENT_START)
    if type_match is None or start == -1:
        raise LLMError("Mock LLM could not find the document in the prompt.")
    start += len(DOCUMENT_START)
    end = prompt.find(DOCUMENT_END, start)
    if end == -1:
        raise LLMError("Mock LLM could not find the end of the document.")
    return type_match.group(1), prompt[start:end].strip("\n")


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
def _find_value(text: str, label_patterns: list[str]) -> Optional[str]:
    """Find ``Label: value`` (optionally ``Label (note): value``) on its own line."""
    labels = "|".join(label_patterns)
    pattern = rf"^[ \t]*(?:{labels})[ \t]*(?:\([^)\n]*\))?[ \t]*:[ \t]*(.+?)[ \t]*$"
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else None


def _to_number(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", raw)
    return float(match.group().replace(",", "")) if match else None


def _whole_if_integral(value: Optional[float]) -> Optional[float]:
    if value is not None and value.is_integer():
        return int(value)
    return value


_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")


def _to_iso_date(raw: Optional[str]) -> Optional[str]:
    """Convert unambiguous date formats to ISO; anything else becomes None."""
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


_STATUS_WORDS = {
    "paid": "paid",
    "unpaid": "unpaid",
    "partially paid": "partially_paid",
    "partial": "partially_paid",
    "overdue": "overdue",
}


def _split_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


# --------------------------------------------------------------------------- #
# Invoice
# --------------------------------------------------------------------------- #
def _parse_line_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = _split_cells(line)
        if len(cells) != 4 or cells[0].lower() == "description":
            continue
        if not cells[0] or set(cells[0]) <= set("-: "):
            continue  # blank or a table separator row
        items.append(
            {
                "description": cells[0],
                "quantity": _whole_if_integral(_to_number(cells[1])),
                "unit_price": _to_number(cells[2]),
                "amount": _to_number(cells[3]),
            }
        )
    return items


def _extract_invoice(text: str) -> dict[str, Any]:
    currency_raw = (_find_value(text, [r"Currency"]) or "").strip()
    status_raw = _find_value(text, [r"Payment\s+Status", r"Status"])
    status_key = re.sub(r"[\s_-]+", " ", status_raw.strip().lower()) if status_raw else ""

    return {
        "invoice_number": _find_value(
            text, [r"Invoice\s+(?:Number|No\.?|#)", r"Invoice\s+ID"]
        ),
        "invoice_date": _to_iso_date(_find_value(text, [r"Invoice\s+Date"])),
        "due_date": _to_iso_date(_find_value(text, [r"Due\s+Date"])),
        "vendor_name": _find_value(text, [r"Vendor", r"From", r"Seller"]),
        "vendor_address": _find_value(text, [r"Vendor\s+Address", r"From\s+Address"]),
        "customer_name": _find_value(text, [r"Bill\s+To", r"Customer", r"Client"]),
        "customer_address": _find_value(
            text, [r"Customer\s+Address", r"Billing\s+Address"]
        ),
        "subtotal": _to_number(_find_value(text, [r"Subtotal"])),
        "tax": _to_number(_find_value(text, [r"Tax", r"GST", r"VAT"])),
        "total_amount": _to_number(
            _find_value(
                text,
                [r"Total(?:\s+Amount)?(?:\s+Due)?", r"Grand\s+Total", r"Amount\s+(?:Due|Payable)"],
            )
        ),
        "currency": currency_raw.upper() if re.fullmatch(r"[A-Za-z]{3}", currency_raw) else None,
        "payment_status": _STATUS_WORDS.get(status_key),
        "line_items": _parse_line_items(text),
    }


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
_SECTION_NAMES = {
    "summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "education": "education",
    "work experience": "work",
    "professional experience": "work",
    "experience": "work",
    "projects": "projects",
    "certifications": "certifications",
    "certificates": "certifications",
}
_INLINE_SECTION_RE = re.compile(r"^\s*([A-Za-z ]+?)\s*:\s*(.+)$")


def _split_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split a resume into (lines before the first section, {section: lines})."""
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        heading = line.strip().rstrip(":").strip().lower()
        inline = _INLINE_SECTION_RE.match(line)
        if heading in _SECTION_NAMES:
            current = _SECTION_NAMES[heading]
            sections.setdefault(current, [])
        elif inline and inline.group(1).strip().lower() in _SECTION_NAMES:
            current = _SECTION_NAMES[inline.group(1).strip().lower()]
            sections.setdefault(current, []).append(inline.group(2).strip())
        elif line.strip():
            (sections[current] if current else preamble).append(line.strip())
    return preamble, sections


def _split_dates(raw: str) -> tuple[Optional[str], Optional[str]]:
    parts = re.split(r"\s+(?:-|–|—|to)\s+", raw.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip() or None, parts[1].strip() or None
    return None, parts[0].strip() or None


def _parse_education(lines: list[str]) -> list[dict[str, Any]]:
    entries = []
    for line in lines:
        cells = _split_cells(line)
        degree = field = start = end = None
        if len(cells) >= 4:
            degree, field, institution = cells[0], cells[1], cells[2]
            start, end = _split_dates(cells[3])
        elif len(cells) == 3:
            degree, institution = cells[0], cells[1]
            start, end = _split_dates(cells[2])
        elif len(cells) == 2:
            degree, institution = cells[0], cells[1]
        else:
            institution = cells[0]
        entries.append(
            {
                "institution": institution,
                "degree": degree,
                "field_of_study": field,
                "start_date": start,
                "end_date": end,
            }
        )
    return entries


def _parse_work(lines: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for line in lines:
        if _BULLET_RE.match(line):
            if jobs:
                jobs[-1]["responsibilities"].append(_BULLET_RE.sub("", line).strip())
            continue
        cells = _split_cells(line)
        if len(cells) < 2:
            continue  # cannot build a valid entry without title and company
        location = start = end = None
        if len(cells) >= 4:
            location = cells[2]
            start, end = _split_dates(cells[3])
        elif len(cells) == 3:
            start, end = _split_dates(cells[2])
        jobs.append(
            {
                "job_title": cells[0],
                "company": cells[1],
                "location": location,
                "start_date": start,
                "end_date": end,
                "responsibilities": [],
            }
        )
    return jobs


def _parse_projects(lines: list[str]) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    for line in lines:
        if _BULLET_RE.match(line):
            if projects:
                bullet = _BULLET_RE.sub("", line).strip()
                previous = projects[-1]["description"]
                projects[-1]["description"] = f"{previous} {bullet}" if previous else bullet
            continue
        cells = _split_cells(line)
        tech_text = re.sub(r"(?i)^(?:technologies|tech|stack)\s*:\s*", "", cells[1]) if len(cells) > 1 else ""
        projects.append(
            {
                "name": cells[0],
                "description": None,
                "technologies": [t.strip() for t in tech_text.split(",") if t.strip()],
            }
        )
    return projects


def _parse_certifications(lines: list[str]) -> list[dict[str, Any]]:
    certifications = []
    for line in lines:
        cells = _split_cells(line)
        certifications.append(
            {
                "name": cells[0],
                "issuer": cells[1] if len(cells) > 1 else None,
                "date": cells[2] if len(cells) > 2 else None,
            }
        )
    return certifications


def _parse_skills(lines: list[str]) -> list[str]:
    skills: list[str] = []
    for line in lines:
        for piece in re.split(r"[,;|]", _BULLET_RE.sub("", line)):
            skill = piece.strip()
            if skill and skill not in skills:
                skills.append(skill)
    return skills


def _extract_resume(text: str) -> dict[str, Any]:
    preamble, sections = _split_sections(text)
    first_line = preamble[0] if preamble else ""
    looks_like_name = bool(first_line) and "@" not in first_line and ":" not in first_line
    email = re.search(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+", text)
    phone = re.search(
        r"(?:phone|mobile|tel)\s*:\s*(\+?[\d(][\d\s().\-]{5,}\d)", text, re.IGNORECASE
    )
    location = _find_value(text, [r"Location"])
    summary_lines = sections.get("summary", [])

    return {
        "full_name": first_line if looks_like_name and len(first_line.split()) <= 6 else None,
        "email": email.group() if email else None,
        "phone": phone.group(1).strip() if phone else None,
        "location": location.split("|")[0].strip() if location else None,
        "summary": " ".join(summary_lines) or None,
        "skills": _parse_skills(sections.get("skills", [])),
        "education": _parse_education(sections.get("education", [])),
        "work_experience": _parse_work(sections.get("work", [])),
        "projects": _parse_projects(sections.get("projects", [])),
        "certifications": _parse_certifications(sections.get("certifications", [])),
    }


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
class MockLLMClient(BaseLLMClient):
    """Rule-based replacement for the LLM. Makes no network calls."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        document_type, document_text = parse_prompt(prompt)
        if document_type == "invoice":
            payload = _extract_invoice(document_text)
        elif document_type == "resume":
            payload = _extract_resume(document_text)
        else:
            raise LLMError(f"Mock LLM does not support document type {document_type!r}.")
        return json.dumps(payload, ensure_ascii=False)
