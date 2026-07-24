"""Unit tests for web-facing sandbox session isolation."""

import os
import threading
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from aero_forge.sandbox import manager as manager_module
from aero_forge.sandbox.manager import SandboxManager, ensure_cargo_in_path


@pytest.fixture
def manager(tmp_path):
    return SandboxManager(base_dir=tmp_path / "sandboxes")


def test_create_session_sandbox_returns_unique_dirs(manager):
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    path_a = manager.create_session_sandbox(session_a)
    path_b = manager.create_session_sandbox(session_b)

    assert path_a != path_b
    assert path_a.is_dir()
    assert path_b.is_dir()
    assert path_a.name == session_a
    assert path_b.name == session_b


def test_session_sandboxes_are_isolated(manager):
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    dir_a = manager.create_session_sandbox(session_a)
    dir_b = manager.create_session_sandbox(session_b)

    (dir_a / "a_only.txt").write_text("a", encoding="utf-8")
    (dir_b / "b_only.txt").write_text("b", encoding="utf-8")

    assert (dir_a / "a_only.txt").read_text() == "a"
    assert not (dir_a / "b_only.txt").exists()
    assert (dir_b / "b_only.txt").read_text() == "b"
    assert not (dir_b / "a_only.txt").exists()


def test_concurrent_sessions_do_not_cross_contaminate(tmp_path):
    manager = SandboxManager(base_dir=tmp_path / "sandboxes")
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    errors = []
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(session_id, marker):
        try:
            sdir = manager.create_session_sandbox(session_id)
            # Synchronize so both threads create dirs before writing.
            barrier.wait(timeout=5)
            (sdir / f"{marker}.txt").write_text(marker, encoding="utf-8")
            # Give the other thread a moment to write its marker.
            time.sleep(0.05)
            with lock:
                results[session_id] = {
                    "dir": sdir,
                    "has_own": (sdir / f"{marker}.txt").is_file(),
                    "has_other": any(
                        (sdir / f"{other}.txt").is_file()
                        for other in ("a", "b")
                        if other != marker
                    ),
                }
        except Exception as exc:  # pragma: no cover
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(session_a, "a")),
        threading.Thread(target=worker, args=(session_b, "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results[session_a]["has_own"]
    assert results[session_b]["has_own"]
    assert not results[session_a]["has_other"]
    assert not results[session_b]["has_other"]


def test_archive_session_sandbox_returns_valid_zip(manager):
    session_id = str(uuid.uuid4())
    sdir = manager.create_session_sandbox(session_id)
    (sdir / "data.txt").write_text("hello", encoding="utf-8")
    nested = sdir / "subdir"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "nested.txt").write_text("world", encoding="utf-8")

    archive_bytes = manager.archive_session_sandbox(session_id)
    assert isinstance(archive_bytes, bytes)
    assert len(archive_bytes) > 0

    with zipfile.ZipFile(BytesIO(archive_bytes), "r") as zf:
        names = zf.namelist()
        assert "data.txt" in names
        assert "subdir/nested.txt" in names


def test_clean_session_sandbox_removes_directory(manager):
    session_id = str(uuid.uuid4())
    sdir = manager.create_session_sandbox(session_id)
    (sdir / "data.txt").write_text("hello", encoding="utf-8")

    assert sdir.is_dir()
    manager.clean_session_sandbox(session_id)
    assert not sdir.exists()
    assert session_id not in manager._sessions


def test_get_session_sandbox_reuses_existing(manager):
    session_id = str(uuid.uuid4())
    first = manager.get_session_sandbox(session_id)
    second = manager.get_session_sandbox(session_id)
    assert first.root == second.root


def test_ensure_cargo_in_path_prepends_cargo_bin(monkeypatch, tmp_path):
    fake_cargo_bin = tmp_path / ".cargo" / "bin"
    fake_cargo_bin.mkdir(parents=True)
    (fake_cargo_bin / "cargo").write_text("#!/bin/sh\necho fake", encoding="utf-8")
    (fake_cargo_bin / "cargo").chmod(0o755)

    original_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(manager_module, "CARGO_BIN_DIR", fake_cargo_bin)
    # Ensure shutil.which sees the bare environment without cargo.
    monkeypatch.setattr(manager_module.shutil, "which", lambda name: None)

    ensure_cargo_in_path()

    assert str(fake_cargo_bin) in os.environ["PATH"]
    assert os.environ["PATH"].startswith(str(fake_cargo_bin))

    os.environ["PATH"] = original_path


def test_ensure_cargo_in_path_does_nothing_when_cargo_present(monkeypatch, tmp_path):
    original_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", "/usr/bin")
    calls = []

    def fake_which(name):
        calls.append(name)
        return "/usr/bin/cargo" if name == "cargo" else None

    monkeypatch.setattr(manager_module.shutil, "which", fake_which)

    ensure_cargo_in_path()

    assert calls == ["cargo"]
    # PATH should be unchanged when cargo is already on PATH.
    assert os.environ["PATH"] == "/usr/bin"

    os.environ["PATH"] = original_path
