import os
import pathlib

import pytest

from tools.file_reader import FileReader


@pytest.fixture
def reader(workspace):
    return FileReader(workspace, max_bytes=5_000)


def test_reads_txt(reader):
    r = reader.read("notes.txt")
    assert r.ok and "agentic AI" in r.content
    assert r.untrusted is True
    assert r.sources[0].kind == "file" and r.sources[0].title == "notes.txt"


def test_reads_csv_with_row_count(reader):
    r = reader.read("expenses.csv")
    assert r.ok and "Housing" in r.content
    assert "CSV data rows (excluding header): 3" in r.content


def test_reads_json(reader):
    r = reader.read("data.json")
    assert r.ok and '"b": [1, 2, 3]' in r.content


def test_reads_markdown(reader):
    r = reader.read("readme.md")
    assert r.ok and "# Title" in r.content


def test_reads_python_source_without_executing(reader, workspace):
    marker = workspace / "executed.flag"
    (workspace / "evil.py").write_text(f"open(r'{marker}', 'w').write('x')\n", encoding="utf-8")
    r = reader.read("evil.py")
    assert r.ok and "open(" in r.content
    assert not marker.exists()


def test_reads_file_in_subdirectory_and_absolute_path_inside_workspace(reader, workspace):
    (workspace / "sub").mkdir()
    (workspace / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    assert reader.read("sub/deep.txt").ok
    assert reader.read(str(workspace / "sub" / "deep.txt")).ok


def test_missing_file_lists_available_files(reader):
    r = reader.read("missing.csv")
    assert not r.ok
    assert "not found" in r.error.lower()
    assert "expenses.csv" in r.error


@pytest.mark.parametrize("name", ["malware.exe", "script.sh", "archive.zip", "noextension", "photo.png"])
def test_unsupported_extension(reader, workspace, name):
    (workspace / name).write_bytes(b"data")
    r = reader.read(name)
    assert not r.ok and "Unsupported file type" in r.error


def test_permission_error(reader, monkeypatch):
    real_open = pathlib.Path.open

    def deny(self, *args, **kwargs):
        if self.name == "notes.txt":
            raise PermissionError("denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", deny)
    r = reader.read("notes.txt")
    assert not r.ok and "Permission denied" in r.error


def test_other_os_error_is_handled(reader, monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(pathlib.Path, "open", boom)
    r = reader.read("notes.txt")
    assert not r.ok and "Could not read" in r.error


@pytest.mark.parametrize("path", [
    "../secret.txt",
    "../../secret.txt",
    "sub/../../secret.txt",
    "..\\secret.txt",
    "/etc/passwd",
    "....//secret.txt",
])
def test_directory_traversal_blocked(reader, workspace, path):
    (workspace.parent / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    r = reader.read(path)
    assert not r.ok
    assert "TOP SECRET" not in (r.content + (r.error or ""))


def test_absolute_path_outside_workspace_blocked(reader, workspace):
    outside = workspace.parent / "secret.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    r = reader.read(str(outside))
    assert not r.ok and "outside the workspace" in r.error or "Access denied" in r.error


def test_symlink_escape_blocked(reader, workspace):
    outside = workspace.parent / "secret.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    r = reader.read("link.txt")
    assert not r.ok and "TOP SECRET" not in r.content


def test_null_byte_and_empty_path(reader):
    assert not reader.read("notes.txt\x00.png").ok
    assert not reader.read("").ok
    assert not reader.read("   ").ok
    assert not reader.read(None).ok  # type: ignore[arg-type]


def test_directory_is_not_readable(reader, workspace):
    (workspace / "folder.txt").mkdir()
    r = reader.read("folder.txt")
    assert not r.ok and "not a regular file" in r.error


def test_binary_content_rejected(reader, workspace):
    (workspace / "bin.txt").write_bytes(b"abc\x00def")
    r = reader.read("bin.txt")
    assert not r.ok and "binary" in r.error


def test_file_size_limit_truncates_on_line_boundary(workspace):
    big = workspace / "big.csv"
    big.write_text("id,value\n" + "".join(f"{i},{i * 2}\n" for i in range(5000)), encoding="utf-8")
    reader = FileReader(workspace, max_bytes=1_000)
    r = reader.read("big.csv")
    assert r.ok
    assert "TRUNCATED" in r.content
    assert len(r.content) < 1_500
    assert "truncated" in r.summary
    body = r.content.split("\n\n", 1)[1]
    assert all(line.count(",") == 1 for line in body.splitlines())  # no half-cut rows


def test_utf8_bom_handled(reader, workspace):
    (workspace / "bom.csv").write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
    r = reader.read("bom.csv")
    assert r.ok and "﻿" not in r.content


def test_error_messages_do_not_leak_absolute_paths(reader, workspace):
    r = reader.read("missing.csv")
    assert str(workspace) not in r.error
