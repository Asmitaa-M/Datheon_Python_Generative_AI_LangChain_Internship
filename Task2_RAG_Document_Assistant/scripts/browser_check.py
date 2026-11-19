"""Drive the running Streamlit app in a real (headless) Chrome and save screenshots.

This is a developer utility used to verify the UI end to end and to produce the
README screenshots. It is not needed to run the application.

Requirements:  pip install playwright   (uses your installed Google Chrome; no browser download)

Usage:
    streamlit run app.py --server.port 8501 &        # in another terminal
    python scripts/browser_check.py --url http://localhost:8501 --out docs/screenshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CHROME_CANDIDATES = ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser")
TITLE = "Document Intelligence – RAG Assistant"


CHAT_MESSAGE = '[data-testid="stChatMessage"]'
REFUSAL = "I couldn't find sufficient information to answer this question in the uploaded documents."


def ask(page: Page, question: str, message_count: int) -> str:
    """Send ``question`` and return the visible text of the newest assistant message.

    ``message_count`` is the total number of chat messages expected afterwards.
    Assertions are scoped to this one message, so text that is always present
    elsewhere on the page (such as the sidebar file list) can never satisfy them.
    """
    box = page.get_by_placeholder("Ask a question about your documents…")
    box.fill(question)
    box.press("Enter")
    messages = page.locator(CHAT_MESSAGE)
    expect(messages).to_have_count(message_count, timeout=30_000)
    answer = messages.last
    expect(answer.get_by_text("Sources (").or_(answer.get_by_text("couldn't find sufficient")).first).to_be_visible(
        timeout=30_000
    )
    answer.scroll_into_view_if_needed()
    return answer


def open_sources(answer) -> str:
    answer.get_by_text("Sources (").first.click()
    answer.page.wait_for_timeout(1000)  # let the expander animation finish
    return answer.inner_text()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    parser.add_argument("--skip-upload", action="store_true", help="only check that previously indexed data is shown")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chrome = next((path for path in CHROME_CANDIDATES if Path(path).exists()), None)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, args=["--no-sandbox"], headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(args.url)
        expect(page.get_by_role("heading", name=TITLE)).to_be_visible(timeout=30_000)

        if args.skip_upload:
            expect(page.get_by_text("machine_learning.pdf").first).to_be_visible(timeout=15_000)
            expect(page.get_by_text("solar_energy.txt").first).to_be_visible(timeout=15_000)
            page.screenshot(path=str(out / "04-after-restart.png"))
            print("OK: previously indexed documents are listed after restart")
            browser.close()
            return 0

        page.screenshot(path=str(out / "01-home.png"))

        page.locator("input[type=file]").set_input_files(
            [str(FIXTURES / "machine_learning.pdf"), str(FIXTURES / "solar_energy.txt"), str(FIXTURES / "corrupted.pdf")]
        )
        page.get_by_role("button", name="Process & index documents").click()
        expect(page.get_by_text("Indexed", exact=False).first).to_be_visible(timeout=60_000)
        expect(page.get_by_text("could not be read as a PDF", exact=False).first).to_be_visible(timeout=15_000)
        page.screenshot(path=str(out / "02-indexed.png"))

        # Scenario 1: answer present -> cited file + page
        answer = ask(page, "What is machine learning according to this document?", message_count=2)
        text = open_sources(answer)
        assert "subfield of artificial intelligence" in text, text
        assert "machine_learning.pdf" in text and "page 1" in text, text
        assert REFUSAL not in text
        page.screenshot(path=str(out / "03-answer-with-sources.png"))

        # Scenario 2: answer absent -> refusal, no sources section
        answer = ask(page, "Who won the football world cup in 2018?", message_count=4)
        text = answer.inner_text()
        assert REFUSAL in text and "Sources (" not in text, text
        page.screenshot(path=str(out / "05-not-found.png"))

        # Scenario 4: question spanning two documents -> both cited
        answer = ask(
            page,
            "What is machine learning and how do solar panels convert sunlight into electricity?",
            message_count=6,
        )
        text = open_sources(answer)
        assert "machine_learning.pdf" in text and "solar_energy.txt" in text, text
        assert REFUSAL not in text
        page.screenshot(path=str(out / "06-two-documents.png"))
        print("OK: upload, indexing, error reporting, cited answer, refusal and multi-document answer verified")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
