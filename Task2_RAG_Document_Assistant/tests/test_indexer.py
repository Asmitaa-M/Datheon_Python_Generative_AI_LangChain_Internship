"""Upload -> index pipeline: validation, duplicates, partial failures."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from src.indexer import DUPLICATE, FAILED, INDEXED, UploadedItem, index_uploads
from src.vector_store import VectorStoreManager


def _item(path: Path, name: str | None = None) -> UploadedItem:
    return UploadedItem(name=name or path.name, data=path.read_bytes())


def test_multiple_files_are_indexed_and_saved(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    results = index_uploads([_item(fixture_paths["ml_pdf"]), _item(fixture_paths["solar_txt"])], store, settings)
    assert [r.status for r in results] == [INDEXED, INDEXED]
    assert all(r.chunks > 0 for r in results)
    assert {doc.source for doc in store.list_documents()} == {"machine_learning.pdf", "solar_energy.txt"}
    assert (settings.upload_dir / "machine_learning.pdf").is_file()


def test_same_file_uploaded_twice_is_not_reprocessed(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    index_uploads([_item(fixture_paths["ml_pdf"])], store, settings)
    count = store.count()
    again = index_uploads([_item(fixture_paths["ml_pdf"])], store, settings)
    assert again[0].status == DUPLICATE
    assert store.count() == count


def test_duplicate_within_one_batch_is_detected(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    results = index_uploads(
        [_item(fixture_paths["solar_txt"]), _item(fixture_paths["solar_txt"], "solar_copy.txt")], store, settings
    )
    assert [r.status for r in results] == [INDEXED, DUPLICATE]


def test_adding_a_second_batch_keeps_existing_documents(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    index_uploads([_item(fixture_paths["ml_pdf"])], store, settings)
    before = store.count()
    index_uploads([_item(fixture_paths["solar_txt"])], store, settings)
    assert store.count() > before
    assert len(store.list_documents()) == 2


def test_bad_files_fail_individually_without_blocking_good_ones(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    items = [
        _item(fixture_paths["corrupted_pdf"]),
        _item(fixture_paths["empty_txt"]),
        _item(fixture_paths["blank_pdf"]),
        UploadedItem("notes.docx", b"not supported"),
        _item(fixture_paths["solar_txt"]),
    ]
    results = index_uploads(items, store, settings)
    statuses = {r.filename: r.status for r in results}
    assert statuses == {
        "corrupted.pdf": FAILED,
        "empty.txt": FAILED,
        "blank_page.pdf": FAILED,
        "notes.docx": FAILED,
        "solar_energy.txt": INDEXED,
    }
    assert "empty" in next(r for r in results if r.filename == "empty.txt").message
    assert "unsupported" in next(r for r in results if r.filename == "notes.docx").message
    assert [doc.source for doc in store.list_documents()] == ["solar_energy.txt"]
    assert not (settings.upload_dir / "corrupted.pdf").exists(), "failed uploads must not be kept on disk"


def test_oversized_file_is_rejected(store: VectorStoreManager, settings: Settings) -> None:
    tiny_limit = Settings(**{**settings.__dict__, "max_upload_mb": 1})
    big = UploadedItem("big.txt", b"x" * (1024 * 1024 + 1))
    assert index_uploads([big], store, tiny_limit)[0].status == FAILED


def test_hostile_filename_is_sanitised(store: VectorStoreManager, settings: Settings) -> None:
    item = UploadedItem("../../etc/evil name.txt", b"Some harmless text about solar energy.")
    result = index_uploads([item], store, settings)[0]
    assert result.status == INDEXED and result.filename == "evil name.txt"
    assert (settings.upload_dir / "evil name.txt").is_file()


def test_same_name_different_content_does_not_overwrite(store: VectorStoreManager, settings: Settings) -> None:
    index_uploads([UploadedItem("notes.txt", b"First version about solar panels.")], store, settings)
    index_uploads([UploadedItem("notes.txt", b"Second version about machine learning.")], store, settings)
    assert len(list(settings.upload_dir.glob("notes*.txt"))) == 2
    assert len(store.list_documents()) == 2


def test_progress_callback_reports_completion(
    store: VectorStoreManager, settings: Settings, fixture_paths: dict[str, Path]
) -> None:
    seen: list[float] = []
    index_uploads([_item(fixture_paths["solar_txt"])], store, settings, progress=lambda f, _m: seen.append(f))
    assert seen[0] == 0.0 and seen[-1] == 1.0
