"""Tests for file handling, input checks, secret redaction and logging."""

from __future__ import annotations

import json
import logging
import socket
from datetime import datetime

import pytest

import utils
from exceptions import EmptyInputError, InputFileError, OutputFileError
from utils import (
    RedactingFormatter,
    build_output_path,
    ensure_non_empty_text,
    read_text_file,
    redact_secrets,
    setup_logging,
    write_json_file,
)

FAKE_KEY = "AIzaSyD-FAKE_KEY_for_tests_0123456789abc"


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def test_ensure_non_empty_text_strips_outer_whitespace():
    assert ensure_non_empty_text("  \n hello world \t\n") == "hello world"


@pytest.mark.parametrize("empty", ["", " ", "\n\n", "\t \r\n", None])
def test_ensure_non_empty_text_rejects_empty_input(empty):
    with pytest.raises(EmptyInputError, match="empty"):
        ensure_non_empty_text(empty)


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #
def test_read_text_file_returns_the_content(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Invoice 1\nTotal: 5", encoding="utf-8")
    assert read_text_file(path) == "Invoice 1\nTotal: 5"
    assert read_text_file(str(path)) == "Invoice 1\nTotal: 5"  # str paths work too


def test_read_text_file_handles_unicode(tmp_path):
    path = tmp_path / "unicode.txt"
    text = "Müller & Söhne – ₹1,200 – 田中 – Ñandú – 😀"
    path.write_text(text, encoding="utf-8")
    assert read_text_file(path) == text


def test_read_text_file_accepts_a_windows_notepad_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfInvoice 7")
    assert read_text_file(path) == "Invoice 7"  # BOM is not part of the text


def test_read_text_file_normalises_windows_line_endings(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line one\r\nline two\r\n")
    assert read_text_file(path) == "line one\nline two\n"


def test_missing_input_file_is_reported(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(InputFileError, match="not found") as caught:
        read_text_file(missing)
    assert "does_not_exist.txt" in str(caught.value)


def test_directory_instead_of_file_is_reported(tmp_path):
    with pytest.raises(InputFileError, match="directory"):
        read_text_file(tmp_path)


def test_non_utf8_file_is_reported_with_a_hint(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(InputFileError, match="UTF-8"):
        read_text_file(path)


# --------------------------------------------------------------------------- #
# Writing JSON files
# --------------------------------------------------------------------------- #
def test_write_json_file_round_trips(tmp_path):
    data = {"name": "Priya", "scores": [1, 2.5, None], "nested": {"ok": True}}
    path = write_json_file(data, tmp_path / "out.json")
    assert path == tmp_path / "out.json"
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_write_json_file_is_pretty_printed_with_trailing_newline(tmp_path):
    path = write_json_file({"a": 1, "b": [1, 2]}, tmp_path / "out.json")
    text = path.read_text(encoding="utf-8")
    assert text == '{\n  "a": 1,\n  "b": [\n    1,\n    2\n  ]\n}\n'


def test_write_json_file_creates_missing_folders(tmp_path):
    path = write_json_file({"a": 1}, tmp_path / "deep" / "er" / "out.json")
    assert path.is_file()


def test_write_json_file_writes_real_utf8_without_bom(tmp_path):
    data = {"vendor": "Müller & Söhne", "currency": "₹", "name": "田中", "emoji": "😀"}
    path = write_json_file(data, tmp_path / "unicode.json")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert "Müller & Söhne" in text and "₹" in text and "田中" in text and "😀" in text
    assert "\\u" not in text
    assert json.loads(text) == data


def test_write_json_file_overwrites_existing_file(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    write_json_file({"new": True}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}


def test_unserialisable_data_is_reported_and_nothing_is_created(tmp_path):
    target = tmp_path / "newdir" / "out.json"
    with pytest.raises(OutputFileError, match="JSON"):
        write_json_file({"bad": {1, 2, 3}}, target)
    assert not target.parent.exists()


def test_unwritable_location_is_reported(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not folder", encoding="utf-8")
    with pytest.raises(OutputFileError, match="Cannot write"):
        write_json_file({"a": 1}, blocker / "out.json")


def test_failed_write_leaves_no_temporary_files(tmp_path, monkeypatch):
    def failing_replace(*_args, **_kwargs):
        raise PermissionError("simulated: file is locked")

    monkeypatch.setattr(utils.os, "replace", failing_replace)
    with pytest.raises(OutputFileError, match="locked"):
        write_json_file({"a": 1}, tmp_path / "out.json")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Output path building
# --------------------------------------------------------------------------- #
NOW = datetime(2026, 11, 16, 0, 12, 34)


def test_build_output_path_is_descriptive(tmp_path):
    path = build_output_path(tmp_path, "invoice", "examples/sample_invoice.txt", now=NOW)
    assert path == tmp_path / "invoice_sample_invoice_20261116_001234.json"


def test_build_output_path_without_a_source_file(tmp_path):
    assert build_output_path(tmp_path, "resume", None, now=NOW).name == "resume_20261116_001234.json"


def test_build_output_path_sanitises_unsafe_names(tmp_path):
    path = build_output_path(tmp_path, "resume", "my résumé (final).txt", now=NOW)
    assert "my_r_sum_final" in path.name
    assert all(char not in path.name for char in " ()é")


def test_build_output_path_cannot_escape_the_output_folder(tmp_path):
    path = build_output_path(tmp_path, "invoice", "../../etc/passwd", now=NOW)
    assert path.parent == tmp_path


def test_build_output_path_never_overwrites_an_existing_file(tmp_path):
    first = build_output_path(tmp_path, "invoice", "a.txt", now=NOW)
    first.write_text("existing", encoding="utf-8")
    second = build_output_path(tmp_path, "invoice", "a.txt", now=NOW)
    assert second != first and second.name.endswith("_1.json")


# --------------------------------------------------------------------------- #
# Secret redaction
# --------------------------------------------------------------------------- #
def test_redact_removes_explicit_secrets():
    assert redact_secrets("token=abc123456 used abc123456", ["abc123456"]) == (
        "token=[REDACTED] used [REDACTED]"
    )


def test_redact_removes_google_style_keys_even_when_unknown():
    assert FAKE_KEY not in redact_secrets(f"failed with {FAKE_KEY} today")


@pytest.mark.parametrize(
    "text",
    [
        "GEMINI_API_KEY=supersecret",
        "api_key: supersecret",
        "https://example.test/v1?key=supersecret&x=1",
        "API-KEY = supersecret",
    ],
)
def test_redact_handles_key_assignments(text):
    assert "supersecret" not in redact_secrets(text)


def test_redact_leaves_ordinary_text_alone():
    text = "Invoice number: INV-2026-0042, total 77,290.59"
    assert redact_secrets(text, [None, "", "ab"]) == text


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_logging_never_prints_the_api_key(capsys):
    setup_logging(verbose=True, secrets=[FAKE_KEY])
    logger = logging.getLogger("test.redaction")
    logger.info("Connecting with key %s", FAKE_KEY)
    try:
        raise RuntimeError(f"request failed for {FAKE_KEY}")
    except RuntimeError:
        logger.exception("Call failed")

    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.err + captured.out
    assert "[REDACTED]" in captured.err
    assert "Connecting with key" in captured.err  # the rest of the message is kept


def test_logs_go_to_stderr_not_stdout(capsys):
    setup_logging()
    logging.getLogger("test.stream").warning("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.err and captured.out == ""


def test_setup_logging_can_be_called_repeatedly_without_duplicate_output(capsys):
    setup_logging()
    setup_logging()
    setup_logging()
    logging.getLogger("test.once").warning("only once")
    assert capsys.readouterr().err.count("only once") == 1


def test_verbose_flag_controls_the_log_level():
    setup_logging(verbose=False)
    assert logging.getLogger().level == logging.INFO
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_redacting_formatter_scrubs_exception_tracebacks():
    formatter = RedactingFormatter("%(message)s", secrets=["TOPSECRET123"])
    try:
        raise ValueError("leaked TOPSECRET123")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", None, sys.exc_info())
    assert "TOPSECRET123" not in formatter.format(record)


# --------------------------------------------------------------------------- #
# Safety net used by the whole suite
# --------------------------------------------------------------------------- #
def test_the_test_suite_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("example.com", 443), timeout=1)
