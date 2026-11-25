"""Command-line interface for the Structured LLM JSON Extractor.

Examples (PowerShell or any shell)::

    python main.py invoice --file examples/sample_invoice.txt --mock
    python main.py resume --file examples/sample_resume.txt --mock
    python main.py invoice --text "Invoice 42 ..." --no-save
    python main.py            # interactive: choose a type, paste text, type END

Exit codes:
    0 success                       3 configuration / API-key problem
    1 unexpected error              4 LLM / API failure
    2 bad input or usage            5 no valid response after retries
                                    6 could not write the output file
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from config import Settings, load_settings
from exceptions import (
    ConfigurationError,
    EmptyInputError,
    ExtractorError,
    InputFileError,
    LLMAuthenticationError,
    LLMError,
    MaxRetriesExceededError,
    OutputFileError,
    UnsupportedDocumentTypeError,
)
from extractor import DocumentExtractor, save_result
from llm_client import create_llm_client
from schemas import resolve_document_type
from utils import (
    build_output_path,
    ensure_non_empty_text,
    read_text_file,
    redact_secrets,
    setup_logging,
)

logger = logging.getLogger("main")

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_BAD_INPUT = 2
EXIT_CONFIG = 3
EXIT_LLM = 4
EXIT_NO_VALID_RESPONSE = 5
EXIT_OUTPUT = 6

END_MARKER = "END"


def _retry_count(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 0 <= number <= 10:
        raise argparse.ArgumentTypeError("must be between 0 and 10")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Extract structured JSON from invoices or resumes using an LLM.",
    )
    parser.add_argument(
        "document_type",
        nargs="?",
        help="Type of document: 'invoice' or 'resume'. Omit to choose interactively.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-f", "--file", help="Path to a UTF-8 .txt file to read.")
    source.add_argument("-t", "--text", help="Document text passed directly.")
    parser.add_argument(
        "-o", "--output", help="Exact path of the JSON file to write."
    )
    parser.add_argument(
        "--output-dir", help="Folder for the auto-named JSON file (default: output/)."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print the JSON but do not save a file."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the offline mock LLM (no API key, no cost).",
    )
    parser.add_argument("--model", help="Gemini model name (overrides GEMINI_MODEL).")
    parser.add_argument(
        "--max-retries",
        type=_retry_count,
        help="Correction attempts after the first response (overrides MAX_RETRIES).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show debug logging."
    )
    return parser


def _configure_console_encoding() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII text cannot crash on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # stream cannot be reconfigured; keep its encoding


def _say(message: str) -> None:
    """Print a user-facing status message to stderr (stdout is for JSON)."""
    print(message, file=sys.stderr)


def prompt_for_document_type(
    input_func: Optional[Callable[[str], str]] = None,
) -> str:
    """Ask the user which kind of document they want to process."""
    input_func = input_func or input  # resolved at call time (patchable)
    _say("Select the document type:\n  1) invoice\n  2) resume")
    choice = input_func("Enter 1 or 2 (or type the name): ").strip()
    return {"1": "invoice", "2": "resume"}.get(choice, choice)


def read_interactive_text(
    input_func: Optional[Callable[[], str]] = None,
) -> str:
    """Read pasted text line by line until a line containing only ``END``."""
    input_func = input_func or input  # resolved at call time (patchable)
    _say(f"Paste the document text. Type {END_MARKER} on its own line to finish.")
    lines: list[str] = []
    while True:
        try:
            line = input_func()
        except EOFError:
            break
        if line.strip() == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines)


def _obtain_document(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    """Return ``(text, source_file_name)`` from --file, --text, or stdin."""
    if args.file:
        return read_text_file(args.file), args.file
    if args.text is not None:
        return args.text, None
    if sys.stdin.isatty():
        return read_interactive_text(), None
    return sys.stdin.read(), None


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    changes: dict[str, object] = {}
    if args.mock:
        changes["mock_mode"] = True
    if args.model:
        changes["model"] = args.model
    if args.max_retries is not None:
        changes["max_retries"] = args.max_retries
    if args.output_dir:
        changes["output_dir"] = Path(args.output_dir)
    return dataclasses.replace(settings, **changes) if changes else settings


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.document_type is None:
        if not sys.stdin.isatty():
            raise UnsupportedDocumentTypeError(
                "No document type given. Use 'invoice' or 'resume'."
            )
        args.document_type = prompt_for_document_type()
    document_type = resolve_document_type(args.document_type)

    text, source_name = _obtain_document(args)
    text = ensure_non_empty_text(text)

    client = create_llm_client(settings)
    extractor = DocumentExtractor(
        client,
        max_retries=settings.max_retries,
        api_max_retries=settings.api_max_retries,
        api_retry_delay=settings.api_retry_delay,
    )
    result = extractor.extract(document_type, text)

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    if not args.no_save:
        destination = (
            Path(args.output)
            if args.output
            else build_output_path(settings.output_dir, document_type, source_name)
        )
        saved_path = save_result(result, destination)
        _say(
            f"Saved validated {document_type} JSON to {saved_path} "
            f"(valid after {result.attempts} attempt(s))."
        )
    return EXIT_OK


def main(
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        env: Environment mapping; when given, ``.env`` is not read. Mainly for
            tests.
    """
    _configure_console_encoding()
    args = build_parser().parse_args(argv)

    secrets: list[Optional[str]] = []
    try:
        settings = _apply_overrides(load_settings(env=env), args)
        secrets.append(settings.api_key)
        setup_logging(args.verbose, secrets)
        return _run(args, settings)
    except (UnsupportedDocumentTypeError, EmptyInputError, InputFileError) as exc:
        _say(f"Input error: {exc}")
        return EXIT_BAD_INPUT
    except (ConfigurationError, LLMAuthenticationError) as exc:
        _say(f"Configuration error: {redact_secrets(str(exc), secrets)}")
        return EXIT_CONFIG
    except LLMError as exc:
        _say(f"LLM error: {redact_secrets(str(exc), secrets)}")
        return EXIT_LLM
    except MaxRetriesExceededError as exc:
        _say(f"Extraction failed: {exc}")
        return EXIT_NO_VALID_RESPONSE
    except OutputFileError as exc:
        _say(f"Output error: {exc}")
        return EXIT_OUTPUT
    except ExtractorError as exc:  # any other expected error
        _say(f"Error: {redact_secrets(str(exc), secrets)}")
        return EXIT_UNEXPECTED
    except KeyboardInterrupt:
        _say("Cancelled.")
        return EXIT_UNEXPECTED
    except Exception as exc:  # unexpected: log it, never hide it
        logger.exception("Unexpected error")
        _say(f"Unexpected error: {type(exc).__name__}: {redact_secrets(str(exc), secrets)}")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
