# Structured LLM Prompt Pipeline & JSON Extractor

A beginner-friendly Python project that turns messy, unstructured text (invoices and resumes) into **validated, structured JSON** using the Google Gemini API, parameterized few-shot prompts and Pydantic schemas. It includes automatic correction retries, a built-in offline **mock mode** (no API key, no cost) and a pytest suite that never touches the network.

## Problem Statement

LLMs are good at reading free-form documents, but their output is unreliable as data: JSON may be malformed, fields may be missing or have the wrong type, and models sometimes *invent* values that are not in the source. Downstream code needs data it can trust.

## Objectives

- Extract invoice and resume information from raw text.
- Force strict JSON output through carefully designed prompts.
- Validate every response against strict Pydantic schemas.
- Detect problems (bad JSON, wrong types, missing or unexpected fields, fabricated values), send the errors back to the model for correction, and stop after a fixed retry limit.
- Keep API code separate from business logic and make everything testable offline.

## Key Features

- **Two document types:** invoice and resume.
- **Parameterized prompt templates** with few-shot examples (a full example and a sparse example that shows `null` for missing fields).
- **Strict Pydantic v2 schemas:** extra fields rejected, no silent type coercion, all keys required (missing data must be an explicit `null`), email/currency/date/number validation.
- **Grounding check against fabrication:** values such as names, emails, phone numbers and amounts must actually appear in the source text, otherwise the response is rejected and corrected.
- **Correction loop:** validation errors plus the original document are sent back to the model; bounded by `MAX_RETRIES`.
- **Separate API retry budget** (exponential back-off) for rate limits, network failures and server errors.
- **Mock mode** (`--mock`): an offline, rule-based extractor that reads the document text; no API key required.
- **CLI:** pass text, load a `.txt` file, or paste text interactively; results are saved as UTF-8 JSON.
- **Secret safety:** the API key is read only from `.env`/environment, hidden from `repr()`, and redacted from logs and error messages.

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ (developed and tested on 3.13) |
| LLM | Google Gemini via the current `google-genai` SDK |
| Validation | Pydantic v2 |
| Configuration | python-dotenv |
| Testing | pytest |
| Logging | Python standard `logging` |

## Architecture Overview

```
main.py (CLI) -> extractor.py (pipeline)
                    |-- prompt_templates.py  build extraction / correction prompts
                    |-- llm_client.py        Gemini (google-genai) or mock client
                    |-- validator.py         parse JSON -> Pydantic -> grounding check
                    `-- schemas.py           Invoice / Resume models
config.py (settings from .env)   utils.py (file I/O, logging, redaction)
exceptions.py (error types)      mock_llm.py (offline mock model)
```

Pipeline: build prompt -> call LLM -> parse JSON -> validate schema -> check values exist in the source -> on any failure, build a correction prompt from the **original prompt** (so the source document is always preserved) plus the previous answer and the error list -> repeat until valid or `1 + MAX_RETRIES` answers have been used.

## Folder Structure

```
structured-llm-json-extractor/
├── main.py              # command-line interface
├── config.py            # settings loaded from environment / .env
├── llm_client.py        # Gemini client + client factory (API layer only)
├── mock_llm.py          # offline mock "LLM" for --mock mode
├── prompt_templates.py  # parameterized prompts and few-shot examples
├── schemas.py           # Pydantic models
├── extractor.py         # retry / correction pipeline, JSON saving
├── validator.py         # JSON parsing, validation, grounding check
├── utils.py             # file I/O, logging, secret redaction
├── exceptions.py        # custom exception hierarchy
├── examples/
│   ├── sample_invoice.txt
│   └── sample_resume.txt
├── output/              # generated JSON files (git-ignored)
├── tests/               # pytest suite (no network, no API key needed)
├── pytest.ini
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation (Windows PowerShell)

```powershell
cd structured-llm-json-extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## API Key Configuration

1. Create a key at <https://aistudio.google.com/apikey>.
2. Copy the template and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

3. Set `GEMINI_API_KEY=your_real_key`. `.env` is git-ignored; never commit it.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | Required unless `--mock` is used |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name; change it to any Gemini model your key can use |
| `LLM_MOCK_MODE` | `false` | `true` forces mock mode |
| `MAX_RETRIES` | `3` | Correction attempts after the first answer (0–10) |
| `API_MAX_RETRIES` | `2` | Retries for rate limits / network errors |
| `API_RETRY_DELAY` | `2.0` | Seconds before the first API retry (doubles each time) |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `OUTPUT_DIR` | `output` | Where JSON files are saved |

## Usage

### Invoice extraction

```powershell
python main.py invoice --file examples\sample_invoice.txt
```

### Resume extraction

```powershell
python main.py resume --file examples\sample_resume.txt
```

### Other input options

```powershell
# Text on the command line
python main.py invoice --text "Invoice Number: A-1 ..."

# Interactive: choose the type, paste text, finish with a line containing END
python main.py

# Choose the output file or folder; or skip saving
python main.py resume --file examples\sample_resume.txt --output results\priya.json
python main.py resume --file examples\sample_resume.txt --output-dir results
python main.py resume --file examples\sample_resume.txt --no-save
```

Logs go to stderr; stdout carries only the JSON. Results are saved as `output\<type>_<source>_<timestamp>.json`.

### Mock mode (no API key, no cost)

```powershell
python main.py invoice --file examples\sample_invoice.txt --mock
python main.py resume  --file examples\sample_resume.txt  --mock
```

The mock is a simple rule-based parser that reads the input text (it is **not** canned output), but it only understands the layout of the files in `examples\`. Use it to try the pipeline and for demos; use the real Gemini client for arbitrary documents.

### Exit codes

`0` success · `1` unexpected error · `2` bad input (empty text, unknown type, missing file) · `3` configuration (missing/invalid API key) · `4` LLM/API failure · `5` no valid response within the retry limit · `6` output file error

## Running the Tests

```powershell
python -m pytest
```

All external API calls are mocked, so no API key is needed and no money is spent.

## Verified Results

Verified in a Linux sandbox (Python 3.13.18, pydantic 2.14.3, google-genai 3.3.0, pytest 9.2.0):

- `python -m compileall` – no syntax errors.
- `python -m pytest` – **466 passed, 0 failed** (also with `GEMINI_API_KEY` unset).
- `python main.py invoice --file examples/sample_invoice.txt --mock` and the resume equivalent both exit with code 0, write a JSON file to `output/`, and the files re-parse as valid UTF-8 JSON.
- Missing key, missing file, empty text and unsupported type each produce a clear message and the documented exit code; a fake API key is redacted from output.
- `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/` and generated `output/*.json` are git-ignored; `.env.example` and `output/.gitkeep` are tracked.

**Not verified:** a live call to the Gemini API (the sandbox cannot reach Google's servers; the Gemini client is tested only against mocked SDK responses) and execution on Windows itself. The code uses `pathlib` and explicit UTF-8 throughout to stay Windows-compatible. The default model name is a configurable setting; if Google retires it, set `GEMINI_MODEL` to a current one.

## Sample Output

Input: `examples/sample_invoice.txt` -> output (mock mode, shortened):

```json
{
  "invoice_number": "INV-2026-0042",
  "invoice_date": "2026-10-05",
  "due_date": "2026-11-04",
  "vendor_name": "Brightline Software Solutions",
  "customer_name": "Nilgiri Tea Traders",
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
      "amount": 45000.0
    }
  ]
}
```

Resume output has `full_name`, `email`, `phone`, `location`, `summary`, `skills`, `education`, `work_experience`, `projects` and `certifications`. Fields absent from the source are `null` (or an empty list for list fields).

## Error Handling

| Situation | Behaviour |
|---|---|
| Missing/placeholder API key | Clear message, exit 3; use `--mock` to run offline |
| Invalid credentials (401/403) | Not retried; message with the key redacted |
| Rate limit (429), network failure, 5xx | Retried with exponential back-off up to `API_MAX_RETRIES`, then reported |
| Invalid JSON, wrong types, missing or unexpected fields, invalid email/date/number | Errors listed per field and sent back for correction |
| Value not found in source (possible fabrication) | Rejected and corrected; the model is told to use `null` |
| Retries exhausted | `MaxRetriesExceededError`, exit 5, last errors shown; nothing is saved |
| Empty input / unsupported type / missing file | Input error, exit 2 |
| File read/write failure | Input/output error with the path; result is validated again before writing |

Markdown code fences around otherwise valid JSON are stripped before parsing (the prompt forbids them, but models sometimes add them anyway).

## Limitations

- The grounding check compares extracted values with the source text; heavily paraphrased or reformatted values can trigger extra correction rounds.
- Invoice arithmetic (subtotal + tax = total) is not cross-checked.
- Only plain-text input is supported (no PDF/image OCR).

## Future Improvements

- PDF and image input, plus batch processing of whole folders.
- Cross-field checks (line items sum to subtotal).
- More document types (receipts, contracts) via a schema/prompt registry.
- Optional LangChain integration and native Gemini structured-output (`response_schema`) mode.
- Async processing and a small web UI.
