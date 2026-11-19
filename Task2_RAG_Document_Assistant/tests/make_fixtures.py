"""Generate the small sample documents used by the tests (and for manual trials).

Run:  python tests/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

ML_PAGES: list[list[str]] = [
    [
        "Introduction to Machine Learning",
        "Machine learning is a subfield of artificial intelligence that enables computers to learn",
        "patterns from data and improve their performance on a task without being explicitly",
        "programmed for every rule. Instead of following fixed instructions, a machine learning",
        "system builds a model from example data and uses that model to make predictions.",
    ],
    [
        "Types of Machine Learning",
        "Supervised learning trains a model on labeled examples, such as emails marked as spam or",
        "not spam. Unsupervised learning finds structure in unlabeled data, for example by grouping",
        "similar customers into clusters. Reinforcement learning trains an agent to maximize reward",
        "through trial and error.",
    ],
    [
        "Overfitting and Evaluation",
        "Overfitting happens when a model memorizes the training data and performs poorly on new data.",
        "Practitioners reduce overfitting with regularization, more training data, and cross-validation.",
        "Model quality is measured on a held-out test set using metrics such as accuracy and recall.",
    ],
]

SOLAR_TEXT = """Solar Energy Overview

Solar panels convert sunlight into electricity using photovoltaic cells made from silicon.
When photons strike the cell, they free electrons and create a direct electric current.
An inverter then converts this direct current into alternating current for use in homes.

Efficiency and Storage

Typical commercial solar panels reach an efficiency of about 20 percent. Batteries store
surplus electricity produced during the day so that it can be used at night.
"""


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(60, y, line)
            y -= 20
        pdf.showPage()
    pdf.save()


def build_fixtures(directory: Path = FIXTURES_DIR) -> dict[str, Path]:
    """Create all fixture files in ``directory`` and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "ml_pdf": directory / "machine_learning.pdf",
        "solar_txt": directory / "solar_energy.txt",
        "empty_txt": directory / "empty.txt",
        "blank_pdf": directory / "blank_page.pdf",
        "corrupted_pdf": directory / "corrupted.pdf",
    }
    _write_pdf(paths["ml_pdf"], ML_PAGES)
    paths["solar_txt"].write_text(SOLAR_TEXT, encoding="utf-8")
    paths["empty_txt"].write_bytes(b"")
    _write_pdf(paths["blank_pdf"], [[]])
    paths["corrupted_pdf"].write_bytes(b"%PDF-1.4\nthis is not really a pdf \x00\x01\x02 garbage")
    return paths


if __name__ == "__main__":
    for label, file_path in build_fixtures().items():
        print(f"{label}: {file_path}")
