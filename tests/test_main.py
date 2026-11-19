"""End-to-end tests of the command-line interface (always with fake LLMs)."""

from __future__ import annotations

import io
import json

import pytest

import main
from exceptions import LLMAuthenticationError, LLMConnectionError, LLMRateLimitError
from schemas import InvoiceData, ResumeData
from helpers import ScriptedLLM, make_invoice_text, make_resume_text

MOCK_ENV = {"LLM_MOCK_MODE": "true"}
FAST_ENV = {**MOCK_ENV, "API_RETRY_DELAY": "0"}


def run_cli(argv, env=None):
    return main.main(argv, env={} if env is None else env)


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(main, "create_llm_client", lambda _settings: llm)


# --------------------------------------------------------------------------- #
# Complete workflows in mock mode
# --------------------------------------------------------------------------- #
def test_invoice_workflow_in_mock_mode_creates_a_valid_json_file(tmp_path, capsys):
    code = run_cli(
        ["invoice", "--file", "examples/sample_invoice.txt", "--output-dir", str(tmp_path)],
        MOCK_ENV,
    )
    captured = capsys.readouterr()
    assert code == main.EXIT_OK

    files = list(tmp_path.glob("invoice_sample_invoice_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert InvoiceData.model_validate(saved)
    assert json.loads(captured.out) == saved  # stdout carries exactly the saved JSON
    assert "Saved validated invoice JSON" in captured.err


def test_resume_workflow_in_mock_mode_creates_a_valid_json_file(tmp_path, capsys):
    code = run_cli(
        ["resume", "-f", "examples/sample_resume.txt", "-o", str(tmp_path / "result.json")],
        MOCK_ENV,
    )
    assert code == main.EXIT_OK
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert ResumeData.model_validate(saved).full_name == "Priya Ramanathan"
    assert json.loads(capsys.readouterr().out) == saved


def test_workflows_need_no_api_key_in_mock_mode(tmp_path):
    assert run_cli(["invoice", "-f", "examples/sample_invoice.txt", "--no-save", "--mock"], {}) == 0


def test_mock_extraction_reflects_the_input_not_hardcoded_output(tmp_path, capsys):
    text = make_invoice_text(
        number="ZX-77", vendor="Orbit Foods", customer="Delta Cafe",
        items=(("Coffee beans", 4, 12.25), ("Milk", 10, 1.5)), tax=6.4, currency="EUR", status="Overdue",
    )
    source = tmp_path / "custom.txt"
    source.write_text(text, encoding="utf-8")

    assert run_cli(["invoice", "-f", str(source), "--no-save"], MOCK_ENV) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["invoice_number"] == "ZX-77"
    assert data["vendor_name"] == "Orbit Foods"
    assert data["currency"] == "EUR" and data["payment_status"] == "overdue"
    assert [i["description"] for i in data["line_items"]] == ["Coffee beans", "Milk"]
    assert data["subtotal"] == pytest.approx(4 * 12.25 + 10 * 1.5)
    assert data["total_amount"] == pytest.approx(64.0 + 6.4)


def test_mock_mode_returns_null_for_information_the_document_lacks(tmp_path, capsys):
    source = tmp_path / "sparse.txt"
    source.write_text(make_invoice_text(due_date=None, tax=None, currency=None, status=None), encoding="utf-8")
    assert run_cli(["invoice", "-f", str(source), "--no-save"], MOCK_ENV) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["due_date"] is None and data["tax"] is None
    assert data["currency"] is None and data["payment_status"] is None


def test_mock_resume_with_missing_contact_details(tmp_path, capsys):
    source = tmp_path / "r.txt"
    source.write_text(make_resume_text(email=None, phone=None, location=None, summary=None), encoding="utf-8")
    assert run_cli(["resume", "-f", str(source), "--no-save"], MOCK_ENV) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["email"] is None and data["phone"] is None
    assert data["full_name"] == "Alex Doe" and data["skills"] == ["Python", "SQL"]


def test_unicode_file_round_trips_through_the_cli(tmp_path, capsys):
    text = make_invoice_text(vendor="Müller & Söhne GmbH", customer="株式会社テスト", items=(("Café crème", 2, 3.5),))
    source = tmp_path / "unicode.txt"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "unicode.json"

    assert run_cli(["invoice", "-f", str(source), "-o", str(output)], MOCK_ENV) == 0
    raw = output.read_bytes().decode("utf-8")
    assert "Müller & Söhne GmbH" in raw and "株式会社テスト" in raw and "Café crème" in raw
    assert json.loads(raw)["customer_name"] == "株式会社テスト"
    capsys.readouterr()


def test_text_can_be_passed_on_the_command_line(capsys):
    text = make_resume_text(name="Sam Lee")
    assert run_cli(["resume", "--text", text, "--no-save"], MOCK_ENV) == 0
    assert json.loads(capsys.readouterr().out)["full_name"] == "Sam Lee"


def test_text_can_be_piped_through_stdin(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "stdin", io.StringIO(make_resume_text(name="Piped Person")))
    assert run_cli(["resume", "--no-save"], MOCK_ENV) == 0
    assert json.loads(capsys.readouterr().out)["full_name"] == "Piped Person"


def test_no_save_flag_writes_nothing(tmp_path, capsys):
    assert run_cli(["invoice", "-f", "examples/sample_invoice.txt", "--no-save", "--output-dir", str(tmp_path)], MOCK_ENV) == 0
    assert list(tmp_path.iterdir()) == []
    capsys.readouterr()


def test_document_type_is_case_insensitive_on_the_command_line(capsys):
    assert run_cli(["INVOICE", "-f", "examples/sample_invoice.txt", "--no-save"], MOCK_ENV) == 0
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# Interactive mode
# --------------------------------------------------------------------------- #
def test_interactive_mode_reads_type_and_text(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "stdin", io.StringIO("ignored"))
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    answers = iter(["2"] + make_resume_text(name="Typed Person").splitlines() + ["END", "unused"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(answers))

    assert run_cli(["--no-save"], MOCK_ENV) == 0
    assert json.loads(capsys.readouterr().out)["full_name"] == "Typed Person"


def test_prompt_for_document_type_accepts_numbers_and_names():
    assert main.prompt_for_document_type(lambda _p: "1") == "invoice"
    assert main.prompt_for_document_type(lambda _p: " 2 ") == "resume"
    assert main.prompt_for_document_type(lambda _p: "Invoice") == "Invoice"


def test_read_interactive_text_stops_at_end_marker_or_eof():
    lines = iter(["first", "second", "END", "never read"])
    assert main.read_interactive_text(lambda: next(lines)) == "first\nsecond"

    def raise_eof():
        raise EOFError

    assert main.read_interactive_text(raise_eof) == ""


def test_missing_type_without_a_terminal_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "stdin", io.StringIO(""))
    assert run_cli(["--no-save"], MOCK_ENV) == main.EXIT_BAD_INPUT
    assert "No document type" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Error handling and exit codes
# --------------------------------------------------------------------------- #
def test_missing_api_key_exits_with_a_helpful_message(capsys):
    code = run_cli(["invoice", "-f", "examples/sample_invoice.txt", "--no-save"], {})
    err = capsys.readouterr().err
    assert code == main.EXIT_CONFIG
    assert "GEMINI_API_KEY" in err and "--mock" in err


def test_missing_input_file(tmp_path, capsys):
    code = run_cli(["invoice", "-f", str(tmp_path / "nope.txt")], MOCK_ENV)
    assert code == main.EXIT_BAD_INPUT
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize("bad_type", ["contract", "receipt", "pdf"])
def test_unsupported_document_type(bad_type, capsys):
    code = run_cli([bad_type, "--text", "something"], MOCK_ENV)
    assert code == main.EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert "Unsupported document type" in err and "invoice, resume" in err


@pytest.mark.parametrize("blank", ["", "   ", "\n\n"])
def test_empty_text_argument(blank, capsys):
    assert run_cli(["invoice", "--text", blank], MOCK_ENV) == main.EXIT_BAD_INPUT
    assert "empty" in capsys.readouterr().err


def test_empty_input_file(tmp_path, capsys):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert run_cli(["resume", "-f", str(empty)], MOCK_ENV) == main.EXIT_BAD_INPUT
    capsys.readouterr()


def test_unsupported_type_is_reported_before_the_input_is_read(tmp_path, capsys):
    code = run_cli(["contract", "-f", str(tmp_path / "missing.txt")], MOCK_ENV)
    assert code == main.EXIT_BAD_INPUT
    assert "Unsupported document type" in capsys.readouterr().err


def test_invalid_environment_value_is_reported(capsys):
    assert run_cli(["invoice", "--text", "x"], {"MAX_RETRIES": "many"}) == main.EXIT_CONFIG
    assert "MAX_RETRIES" in capsys.readouterr().err


def test_invalid_command_line_values_exit_with_usage_error():
    with pytest.raises(SystemExit) as caught:
        main.build_parser().parse_args(["invoice", "--max-retries", "99"])
    assert caught.value.code == 2


def test_file_and_text_cannot_be_combined():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["invoice", "--file", "a.txt", "--text", "b"])


def test_retry_exhaustion_exits_with_its_own_code(monkeypatch, tmp_path, capsys):
    llm = ScriptedLLM(*(["not json"] * 3))
    use_llm(monkeypatch, llm)
    code = run_cli(["invoice", "--text", "Invoice 1", "--max-retries", "2", "-o", str(tmp_path / "x.json")], FAST_ENV)
    err = capsys.readouterr().err
    assert code == main.EXIT_NO_VALID_RESPONSE
    assert "3 attempt(s)" in err and "not valid JSON" in err
    assert llm.call_count == 3
    assert not (tmp_path / "x.json").exists()  # nothing invalid is saved


def test_cli_recovers_from_a_bad_first_response(monkeypatch, tmp_path, capsys, resume_payload, resume_text):
    use_llm(monkeypatch, ScriptedLLM("oops", resume_payload))
    output = tmp_path / "out.json"
    assert run_cli(["resume", "--text", resume_text, "-o", str(output)], FAST_ENV) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["full_name"] == "Priya Ramanathan"
    assert "valid after 2 attempt(s)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error,expected_code,expected_text",
    [
        (LLMAuthenticationError("Gemini rejected the API key."), main.EXIT_CONFIG, "rejected the API key"),
        (LLMConnectionError("Could not reach the Gemini API"), main.EXIT_LLM, "Could not reach"),
        (LLMRateLimitError("Gemini rate limit or quota reached"), main.EXIT_LLM, "rate limit"),
    ],
)
def test_api_failures_map_to_exit_codes(monkeypatch, capsys, error, expected_code, expected_text):
    use_llm(monkeypatch, ScriptedLLM(*[error] * 5))
    code = run_cli(["invoice", "--text", "Invoice 1"], FAST_ENV)
    assert code == expected_code
    assert expected_text in capsys.readouterr().err


def test_unwritable_output_location_exits_with_output_error(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code = run_cli(["invoice", "-f", "examples/sample_invoice.txt", "-o", str(blocker / "o.json")], MOCK_ENV)
    assert code == main.EXIT_OUTPUT
    assert "Cannot write output file" in capsys.readouterr().err


def test_unexpected_bugs_are_reported_not_hidden(monkeypatch, capsys):
    use_llm(monkeypatch, ScriptedLLM(ZeroDivisionError("boom")))
    assert run_cli(["invoice", "--text", "Invoice 1"], FAST_ENV) == main.EXIT_UNEXPECTED
    err = capsys.readouterr().err
    assert "Unexpected error: ZeroDivisionError" in err and "Traceback" in err


def test_api_key_is_never_printed_even_when_an_error_contains_it(monkeypatch, capsys):
    secret = "AIzaSyD-NOT_A_REAL_KEY_but_looks_like_one_123"
    use_llm(monkeypatch, ScriptedLLM(LLMAuthenticationError(f"bad key {secret}")))
    code = run_cli(["invoice", "--text", "Invoice 1"], {"GEMINI_API_KEY": secret, "API_RETRY_DELAY": "0"})
    captured = capsys.readouterr()
    assert code == main.EXIT_CONFIG
    assert secret not in captured.out + captured.err
    assert "[REDACTED]" in captured.err


def test_command_line_overrides_win_over_environment(monkeypatch, capsys):
    seen = {}

    def fake_factory(settings):
        seen["settings"] = settings
        return ScriptedLLM(make_resume_payload())

    def make_resume_payload():
        return {k: None for k in ResumeData.model_fields} | {
            "skills": [], "education": [], "work_experience": [], "projects": [], "certifications": [],
        }

    monkeypatch.setattr(main, "create_llm_client", fake_factory)
    run_cli(["resume", "--text", "Some text", "--model", "gemini-cli", "--max-retries", "7", "--no-save"],
            {"GEMINI_MODEL": "from-env", "MAX_RETRIES": "1"})
    assert seen["settings"].model == "gemini-cli"
    assert seen["settings"].max_retries == 7
    capsys.readouterr()
