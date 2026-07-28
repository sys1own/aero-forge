"""Tests for the workspace export UI endpoint and self-heal integration."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from aero_forge.server import _session_dir


def _post_json(base_url, client, path, payload):
    body = json.dumps(payload).encode("utf-8")
    client.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json"},
    )
    resp = client.getresponse()
    data = resp.read()
    return resp.status, data


def _get(base_url, client, path):
    client.request("GET", path)
    resp = client.getresponse()
    return resp.status, resp.read()


def _make_client(base_url):
    from http.client import HTTPConnection
    return HTTPConnection(base_url.replace("http://", "").replace("https://", ""))


@pytest.fixture
def server():
    import subprocess
    import time

    base_url = "http://localhost:39051"
    proc = subprocess.Popen(
        ["python", "-m", "aero_forge.server", "--port", "39051"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            client = _make_client(base_url)
            client.request("GET", "/")
            client.getresponse()
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("Server did not start")
    yield base_url
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _save_file(base_url, session_id, path, content):
    client = _make_client(base_url)
    status, _ = _post_json(
        base_url,
        client,
        "/api/save-file",
        {"session_id": session_id, "path": path, "content": content},
    )
    assert status == 200


def _export(base_url, session_id, options):
    client = _make_client(base_url)
    status, data = _post_json(
        base_url,
        client,
        "/api/workspace/export",
        {"session_id": session_id, "options": options},
    )
    return status, data


def test_export_pure_target_source(server, tmp_path):
    session_id = "test-export-pure"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    status, data = _export(server, session_id, {"pure_target": True})
    assert status == 200
    assert zipfile.is_zipfile(io.BytesIO(data))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "main.py" in zf.namelist()


def test_export_includes_native_crate(server):
    session_id = "test-export-native"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    status, data = _export(
        server,
        session_id,
        {"pure_target": True, "include_native_crate": True},
    )
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "main.py" in names
        assert "crates/native_core/Cargo.toml" in names
        assert "pyproject.toml" in names


def test_export_includes_wavefront_runtime(server):
    session_id = "test-export-wavefront"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    status, data = _export(
        server,
        session_id,
        {"pure_target": True, "include_wavefront_runtime": True},
    )
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "main.py" in names
        assert "crates/aero_core/Cargo.toml" in names
        assert "crates/aero_core/src/wavefront.rs" in names


def test_export_standalone_aeroc(server):
    session_id = "test-export-aeroc"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    status, data = _export(
        server,
        session_id,
        {"pure_target": True, "standalone_aeroc": True},
    )
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert any(name.endswith(".aerozip") for name in names)


def test_export_all_options(server):
    session_id = "test-export-all"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    status, data = _export(
        server,
        session_id,
        {
            "pure_target": True,
            "include_native_crate": True,
            "include_wavefront_runtime": True,
            "standalone_aeroc": True,
        },
    )
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "main.py" in names
        assert "crates/native_core/Cargo.toml" in names
        assert "crates/aero_core/Cargo.toml" in names
        assert any(name.endswith(".aerozip") for name in names)


def test_download_aeroc_is_raw_binary_not_zip(server):
    """/api/workspace/download-aeroc returns the compiled binary IR container."""
    session_id = "test-download-aeroc"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    client = _make_client(server)
    status, data = _post_json(
        server, client, "/api/workspace/download-aeroc", {"session_id": session_id}
    )
    assert status == 200
    assert data.startswith(b"AEROFOG\x00")
    assert not data.startswith(b"PK")


def test_export_scaffold_is_zip(server):
    """/api/workspace/export-scaffold returns a .aerozip zip bundle."""
    session_id = "test-export-scaffold"
    _save_file(server, session_id, "main.py", "print('hello')\n")
    client = _make_client(server)
    status, data = _post_json(
        server, client, "/api/workspace/export-scaffold", {"session_id": session_id}
    )
    assert status == 200
    assert zipfile.is_zipfile(io.BytesIO(data))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "main.py" in names
        assert "aeroc/aero_core/Cargo.toml" in names
        assert "pyproject.toml" in names
