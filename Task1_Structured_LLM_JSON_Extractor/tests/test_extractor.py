"""Tests for the extraction pipeline: retries, corrections, API failures, saving."""

from __future__ import annotations

import copy
import json

import pytest

from exceptions import (
    EmptyInputError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    MaxRetriesExceededError,
    OutputFileError,
    SchemaValidationError,
    UnsupportedDocumentTypeError,
)
from extractor import DocumentExtractor, ExtractionResult, save_result
from mock_llm import MockLLMClient
from prompt_templates import DOCUMENT_START
from schemas import InvoiceData, ResumeData
from helpers import ScriptedLLM


def make_extractor(llm, **kwargs):
    """Extractor whose backoff never really sleeps. Returns (extractor, sleeps)."""
    sleeps: list[float] = []
    kwargs.setdefault("api_retry_delay", 1.0)
    return DocumentExtractor(llm, sleep=sleeps.append, **kwargs), sleeps


# --------------------------------------------------------------------------- #
# Successful extraction (scripted LLM)
# --------------------------------------------------------------------------- #
def test_valid_invoice_extraction(invoice_payload, invoice_text):
    llm = ScriptedLLM(invoice_payload)
    result = make_extractor(llm)[0].extract("invoice", invoice_text)

    assert isinstance(result, ExtractionResult)
    assert isinstance(result.data, InvoiceData)
    assert result.document_type == "invoice"
    assert result.attempts == 1
    assert llm.call_count == 1
    assert result.to_dict()["total_amount"] == invoice_payload["total_amount"]


def test_valid_resume_extraction(resume_payload, resume_text):
    llm = ScriptedLLM(resume_payload)
    result = make_extractor(llm)[0].extract("resume", resume_text)

    assert isinstance(result.data, ResumeData)
    assert result.data.email == resume_payload["email"]
    assert result.attempts == 1


def test_the_document_is_what_gets_sent_to_the_llm(invoice_payload, invoice_text):
    llm = ScriptedLLM(invoice_payload)
    make_extractor(llm)[0].extract("invoice", invoice_text)
    assert invoice_text.strip() in llm.prompts[0]
    assert "DOCUMENT_TYPE: invoice" in llm.prompts[0]


def test_document_type_is_case_insensitive(invoice_payload, invoice_text):
    result = make_extractor(ScriptedLLM(invoice_payload))[0].extract(" INVOICE ", invoice_text)
    assert result.document_type == "invoice"


def test_missing_optional_information_stays_null(invoice_payload):
    """When the document lacks a value, the validated result keeps null."""
    source = "Invoice INV-2026-0042 from Brightline Software Solutions"
    sparse = {name: None for name in invoice_payload}
    sparse.update(
        invoice_number="INV-2026-0042",
        vendor_name="Brightline Software Solutions",
        line_items=[],
    )
    result = make_extractor(ScriptedLLM(sparse))[0].extract("invoice", source)
    data = result.to_dict()
    assert data["total_amount"] is None and data["tax"] is None
    assert data["line_items"] == []
    assert data["vendor_name"] == "Brightline Software Solutions"


# --------------------------------------------------------------------------- #
# Full pipeline with the offline mock client
# --------------------------------------------------------------------------- #
def test_invoice_pipeline_with_mock_client(invoice_text):
    result = make_extractor(MockLLMClient())[0].extract("invoice", invoice_text)
    data = result.to_dict()
    assert data["invoice_number"] in invoice_text
    table_rows = [line for line in invoice_text.splitlines() if "|" in line]
    assert len(data["line_items"]) == len(table_rows) - 1  # minus the header row
    # Independent consistency check: the line items add up to the printed subtotal.
    assert sum(item["amount"] for item in data["line_items"]) == pytest.approx(data["subtotal"])
    assert result.attempts == 1


def test_resume_pipeline_with_mock_client(resume_text):
    result = make_extractor(MockLLMClient())[0].extract("resume", resume_text)
    data = result.to_dict()
    assert data["email"] in resume_text
    assert [job["company"] for job in data["work_experience"]] == [
        "Tamarind Technologies",
        "Kaveri Labs",
    ]
    assert data["projects"][0]["technologies"] == ["Python", "Flask", "SQLite"]


# --------------------------------------------------------------------------- #
# Input validation happens before any LLM call
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("empty", ["", "   ", "\n\n\t  \n", None])
def test_empty_document_is_rejected_without_calling_the_llm(empty):
    llm = ScriptedLLM()
    with pytest.raises(EmptyInputError):
        make_extractor(llm)[0].extract("invoice", empty)
    assert llm.call_count == 0


@pytest.mark.parametrize("bad_type", ["contract", "", "receipt", None])
def test_unsupported_document_type_is_rejected_without_calling_the_llm(bad_type):
    llm = ScriptedLLM()
    with pytest.raises(UnsupportedDocumentTypeError):
        make_extractor(llm)[0].extract(bad_type, "some text")
    assert llm.call_count == 0


def test_negative_retry_limits_are_rejected():
    with pytest.raises(ValueError):
        DocumentExtractor(ScriptedLLM(), max_retries=-1)
    with pytest.raises(ValueError):
        DocumentExtractor(ScriptedLLM(), api_max_retries=-1)


# --------------------------------------------------------------------------- #
# Retry after malformed JSON
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "malformed,expected_message",
    [
        ("", "The response was empty"),
        ("I'm sorry, I cannot do that.", "is not valid JSON"),
        ('{"invoice_number": "x"', "is not valid JSON"),
        ("```json\n{broken}\n```", "is not valid JSON"),
        ("[]", "top-level JSON value must be an object"),
    ],
)
def test_retry_after_json_parsing_failure(
    malformed, expected_message, invoice_payload, invoice_text
):
    llm = ScriptedLLM(malformed, invoice_payload)
    result = make_extractor(llm)[0].extract("invoice", invoice_text)

    assert result.attempts == 2
    assert llm.call_count == 2
    second_prompt = llm.prompts[1]
    assert "CORRECTION REQUIRED" in second_prompt
    assert expected_message in second_prompt
    assert invoice_text.strip() in second_prompt  # original document preserved


def test_correction_prompt_contains_the_previous_bad_response(invoice_payload, invoice_text):
    llm = ScriptedLLM("this is not json", invoice_payload)
    make_extractor(llm)[0].extract("invoice", invoice_text)
    assert "this is not json" in llm.prompts[1]


# --------------------------------------------------------------------------- #
# Retry after Pydantic validation failure
# --------------------------------------------------------------------------- #
def test_retry_after_pydantic_validation_failure(invoice_payload, invoice_text):
    bad = copy.deepcopy(invoice_payload)
    bad["currency"] = "rupees"
    bad["total_amount"] = "77,290.59"
    llm = ScriptedLLM(bad, invoice_payload)
    result = make_extractor(llm)[0].extract("invoice", invoice_text)

    assert result.attempts == 2
    assert result.data.currency == "INR"
    assert "currency: invalid value" in llm.prompts[1]
    assert "total_amount: invalid type" in llm.prompts[1]
    assert invoice_text.strip() in llm.prompts[1]


def test_retry_after_invalid_email(resume_payload, resume_text):
    bad = copy.deepcopy(resume_payload)
    bad["email"] = "priya.ramanathan(at)example.com"
    llm = ScriptedLLM(bad, resume_payload)
    result = make_extractor(llm)[0].extract("resume", resume_text)
    assert result.attempts == 2
    assert "email: invalid value" in llm.prompts[1]


def test_unexpected_fields_trigger_a_correction(invoice_payload, invoice_text):
    extra = {**invoice_payload, "notes": "paid by cheque", "line_items": [
        {**invoice_payload["line_items"][0], "sku": "X-1"}
    ]}
    llm = ScriptedLLM(extra, invoice_payload)
    result = make_extractor(llm)[0].extract("invoice", invoice_text)

    assert result.attempts == 2
    assert "notes: unexpected field" in llm.prompts[1]
    assert "line_items.0.sku: unexpected field" in llm.prompts[1]
    assert "notes" not in result.to_dict()


def test_incorrect_data_types_trigger_a_correction(resume_payload, resume_text):
    wrong_types = {**resume_payload, "skills": "Python, SQL, Docker", "education": {"x": 1}}
    llm = ScriptedLLM(wrong_types, resume_payload)
    result = make_extractor(llm)[0].extract("resume", resume_text)

    assert result.attempts == 2
    assert "skills: invalid type" in llm.prompts[1]
    assert "education: invalid type" in llm.prompts[1]


def test_several_consecutive_failures_then_success(invoice_payload, invoice_text):
    llm = ScriptedLLM("nope", {"invoice_number": "x"}, "{", invoice_payload)
    result = make_extractor(llm, max_retries=3)[0].extract("invoice", invoice_text)
    assert result.attempts == 4


def test_prompts_do_not_snowball_across_retries(invoice_payload, invoice_text):
    llm = ScriptedLLM("bad one", "bad two", "bad three", invoice_payload)
    make_extractor(llm, max_retries=3)[0].extract("invoice", invoice_text)
    sizes = [len(p) for p in llm.prompts[1:]]
    assert max(sizes) - min(sizes) < 200  # each correction is built from the original
    assert all(p.count("CORRECTION REQUIRED") == 1 for p in llm.prompts[1:])


# --------------------------------------------------------------------------- #
# Maximum retry limit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_retries", [0, 1, 3, 5])
def test_stops_at_the_configured_retry_limit(max_retries, invoice_text):
    llm = ScriptedLLM(*(["still not json"] * (max_retries + 1)))
    with pytest.raises(MaxRetriesExceededError) as caught:
        make_extractor(llm, max_retries=max_retries)[0].extract("invoice", invoice_text)

    assert llm.call_count == max_retries + 1  # one more call would fail the test
    assert caught.value.attempts == max_retries + 1
    assert caught.value.last_errors
    assert "not valid JSON" in str(caught.value)


def test_retry_exhaustion_reports_the_latest_validation_problems(invoice_payload, invoice_text):
    invalid = {**invoice_payload, "tax": -1}
    llm = ScriptedLLM("garbage", invalid, invalid)
    with pytest.raises(MaxRetriesExceededError) as caught:
        make_extractor(llm, max_retries=2)[0].extract("invoice", invoice_text)
    assert any(line.startswith("tax:") for line in caught.value.last_errors)


def test_default_retry_limit_is_bounded(invoice_text):
    llm = ScriptedLLM(*(["bad"] * 4))
    with pytest.raises(MaxRetriesExceededError):
        DocumentExtractor(llm, sleep=lambda _s: None).extract("invoice", invoice_text)
    assert llm.call_count == 4  # 1 attempt + 3 default corrections


# --------------------------------------------------------------------------- #
# Prevention of fabricated information
# --------------------------------------------------------------------------- #
def test_fabricated_value_is_rejected_then_corrected_to_null(resume_payload, resume_text):
    fabricated = {**resume_payload, "email": "made.up@invented-company.com"}
    corrected = {**resume_payload, "email": None}
    llm = ScriptedLLM(fabricated, corrected)
    result = make_extractor(llm)[0].extract("resume", resume_text)

    assert result.attempts == 2
    assert result.data.email is None
    assert "email:" in llm.prompts[1]
    assert "does not appear in the source document" in llm.prompts[1]
    assert "use null" in llm.prompts[1]


def test_llm_that_keeps_fabricating_never_gets_its_data_accepted(invoice_payload, invoice_text):
    fabricated = {**invoice_payload, "invoice_number": "INV-FAKE-1", "total_amount": 999.99}
    llm = ScriptedLLM(fabricated, fabricated, fabricated)
    with pytest.raises(MaxRetriesExceededError) as caught:
        make_extractor(llm, max_retries=2)[0].extract("invoice", invoice_text)
    fields = {line.split(":")[0] for line in caught.value.last_errors}
    assert fields == {"invoice_number", "total_amount"}


def test_document_without_an_email_cannot_yield_one(resume_payload):
    source = "Priya Ramanathan\nPhone: +91 98765 43210\nSKILLS\nPython"
    with_email = {**resume_payload}  # still has an e-mail the document never mentions
    without_email = {**resume_payload, "email": None}
    llm = ScriptedLLM(with_email, without_email)
    result = make_extractor(llm)[0].extract("resume", source)
    assert result.data.email is None


def test_derived_totals_are_not_accepted_as_facts(invoice_payload):
    source = "INV-2026-0042 Brightline Software Solutions Nilgiri Tea Traders\nSubtotal: 65,500.50\nTax: 11,790.09"
    calculated = {**invoice_payload, "line_items": []}  # total 77290.59 is NOT in the text
    corrected = {**calculated, "total_amount": None}
    llm = ScriptedLLM(calculated, corrected)
    result = make_extractor(llm)[0].extract("invoice", source)
    assert result.data.total_amount is None
    assert result.data.subtotal == 65500.5


# --------------------------------------------------------------------------- #
# API failure handling
# --------------------------------------------------------------------------- #
def test_transient_errors_are_retried_with_exponential_backoff(invoice_payload, invoice_text):
    llm = ScriptedLLM(LLMConnectionError("down"), LLMRateLimitError("slow down"), invoice_payload)
    extractor, sleeps = make_extractor(llm, api_max_retries=2, api_retry_delay=1.5)
    result = extractor.extract("invoice", invoice_text)

    assert result.attempts == 1  # API retries do not count as correction attempts
    assert llm.call_count == 3
    assert sleeps == [1.5, 3.0]


@pytest.mark.parametrize("error", [LLMConnectionError, LLMRateLimitError, LLMServerError])
def test_persistent_transient_errors_are_raised_after_the_limit(error, invoice_text):
    llm = ScriptedLLM(*[error("still failing")] * 3)
    extractor, sleeps = make_extractor(llm, api_max_retries=2)
    with pytest.raises(error):
        extractor.extract("invoice", invoice_text)
    assert llm.call_count == 3
    assert len(sleeps) == 2


def test_zero_api_retries_means_a_single_call(invoice_text):
    llm = ScriptedLLM(LLMRateLimitError("quota"))
    extractor, sleeps = make_extractor(llm, api_max_retries=0)
    with pytest.raises(LLMRateLimitError):
        extractor.extract("invoice", invoice_text)
    assert llm.call_count == 1 and sleeps == []


def test_authentication_errors_are_never_retried(invoice_text):
    llm = ScriptedLLM(LLMAuthenticationError("bad key"))
    extractor, sleeps = make_extractor(llm, api_max_retries=5)
    with pytest.raises(LLMAuthenticationError):
        extractor.extract("invoice", invoice_text)
    assert llm.call_count == 1 and sleeps == []


def test_generic_llm_errors_are_not_retried(invoice_text):
    llm = ScriptedLLM(LLMError("empty response"))
    extractor, sleeps = make_extractor(llm)
    with pytest.raises(LLMError):
        extractor.extract("invoice", invoice_text)
    assert llm.call_count == 1 and sleeps == []


def test_unexpected_exceptions_propagate_instead_of_being_swallowed(invoice_text):
    llm = ScriptedLLM(ZeroDivisionError("bug"))
    with pytest.raises(ZeroDivisionError):
        make_extractor(llm)[0].extract("invoice", invoice_text)


def test_api_retries_and_correction_retries_have_separate_budgets(invoice_payload, invoice_text):
    llm = ScriptedLLM("bad json", LLMRateLimitError("slow"), invoice_payload)
    extractor, sleeps = make_extractor(llm, max_retries=1, api_max_retries=1)
    result = extractor.extract("invoice", invoice_text)
    assert result.attempts == 2
    assert len(sleeps) == 1


# --------------------------------------------------------------------------- #
# Saving results
# --------------------------------------------------------------------------- #
def test_successful_json_file_generation(tmp_path, invoice_payload, invoice_text):
    result = make_extractor(ScriptedLLM(invoice_payload))[0].extract("invoice", invoice_text)
    path = save_result(result, tmp_path / "nested" / "invoice.json")

    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == result.to_dict()
    assert InvoiceData.model_validate(reloaded)  # the saved file is itself valid
    assert list(reloaded) == list(InvoiceData.model_fields)  # consistent key order
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert "\n  " in path.read_text(encoding="utf-8")  # human-readable indentation


def test_saved_resume_round_trips(tmp_path, resume_payload, resume_text):
    result = make_extractor(ScriptedLLM(resume_payload))[0].extract("resume", resume_text)
    path = save_result(result, tmp_path / "resume.json")
    assert ResumeData.model_validate(json.loads(path.read_text(encoding="utf-8"))) == result.data


def test_unicode_is_written_as_real_utf8_not_escapes(tmp_path, invoice_payload):
    invoice_payload["vendor_name"] = "Müller & Söhne GmbH"
    invoice_payload["customer_name"] = "株式会社テスト"
    invoice_payload["line_items"][0]["description"] = "Café crème ₹500 – Größe"
    source = "Müller & Söhne GmbH 株式会社テスト INV-2026-0042 Café crème ₹500 – Größe 3 15000 45000"
    result = make_extractor(ScriptedLLM(invoice_payload))[0].extract("invoice", source + " 65500.50 11790.09 77290.59 1 8500.50")
    path = save_result(result, tmp_path / "unicode.json")

    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")  # must be valid UTF-8
    assert "Müller & Söhne GmbH" in raw_text
    assert "株式会社テスト" in raw_text
    assert "₹500" in raw_text
    assert "\\u" not in raw_text
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")  # no BOM
    assert json.loads(raw_text)["customer_name"] == "株式会社テスト"


def test_invalid_data_is_never_written_to_disk(tmp_path, invoice_payload):
    valid = InvoiceData.model_validate(invoice_payload)
    corrupted = valid.model_copy(update={"currency": "not-a-code", "total_amount": -5})
    result = ExtractionResult("invoice", corrupted, attempts=1)
    target = tmp_path / "bad.json"
    with pytest.raises(SchemaValidationError) as caught:
        save_result(result, target)
    assert not target.exists()
    assert {line.split(":")[0] for line in caught.value.errors} == {"currency", "total_amount"}


def test_save_failure_is_reported_clearly(tmp_path, invoice_payload):
    result = ExtractionResult("invoice", InvoiceData.model_validate(invoice_payload), 1)
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a folder", encoding="utf-8")
    with pytest.raises(OutputFileError):
        save_result(result, blocker / "out.json")


def test_saving_overwrites_an_existing_file_atomically(tmp_path, invoice_payload):
    path = tmp_path / "out.json"
    path.write_text("old content", encoding="utf-8")
    result = ExtractionResult("invoice", InvoiceData.model_validate(invoice_payload), 1)
    save_result(result, path)
    assert json.loads(path.read_text(encoding="utf-8"))["invoice_number"] == "INV-2026-0042"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]  # no temp files left


def test_prompt_marker_is_present_for_every_attempt(invoice_payload, invoice_text):
    llm = ScriptedLLM("bad", invoice_payload)
    make_extractor(llm)[0].extract("invoice", invoice_text)
    assert all(p.count(DOCUMENT_START) == 1 for p in llm.prompts)
