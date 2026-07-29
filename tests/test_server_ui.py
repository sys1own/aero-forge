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


def _post_bytes(base_url, client, path, data, content_type="application/octet-stream", query=""):
    full_path = f"{path}?{query}" if query else path
    client.request("POST", full_path, body=data, headers={"Content-Type": content_type})
    resp = client.getresponse()
    return resp.status, resp.read()


def _post_multipart(base_url, client, path, filename, data, query=""):
    boundary = "----AeroForgeTestBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8")
    body += data
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    full_path = f"{path}?{query}" if query else path
    client.request(
        "POST",
        full_path,
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = client.getresponse()
    return resp.status, resp.read()


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


def test_upload_zip_returns_populated_commands(server):
    """POST /api/upload extracts a zip and returns standardized runnable commands."""
    session_id = "test-upload-commands"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("main.py", 'if __name__ == "__main__":\n    print("ok")\n')
        zf.writestr("tests/test_ok.py", "def test_ok(): pass")

    client = _make_client(server)
    status, data = _post_multipart(
        server,
        client,
        "/api/upload",
        "project.zip",
        zip_buf.getvalue(),
        query=f"session_id={session_id}&target_path=.",
    )
    assert status == 200, data.decode("utf-8", errors="ignore")
    payload = json.loads(data)
    assert payload["status"] == "success"
    commands = payload["commands"]
    assert any(c["cmd"] == "python main.py" and c["category"] == "run" for c in commands)
    assert any(c["cmd"] == "pytest" and c["category"] == "test" for c in commands)


def test_unpack_aeroc_returns_populated_commands(server, tmp_path):
    """POST /api/unpack extracts a .aeroc container and returns runnable commands."""
    from aero_forge.builder.aeroc_compiler import compile_directory_to_aeroc

    workspace = tmp_path / "aeroc_src"
    workspace.mkdir()
    (workspace / "main.py").write_text('if __name__ == "__main__":\n    print("ok")\n')
    aeroc = tmp_path / "workspace.aeroc"
    compile_directory_to_aeroc(workspace, aeroc)

    session_id = "test-unpack-commands"
    client = _make_client(server)
    status, data = _post_bytes(
        server,
        client,
        "/api/unpack",
        aeroc.read_bytes(),
        query=f"session_id={session_id}&target_path=.",
    )
    assert status == 200, data.decode("utf-8", errors="ignore")
    payload = json.loads(data)
    assert payload["status"] == "success"
    commands = payload["commands"]
    assert any(c["cmd"] == "python main.py" for c in commands)


def test_generate_returns_command_list(server):
    """POST /api/generate returns a standardized command list even when generation fails."""
    session_id = "test-generate-commands"
    _save_file(server, session_id, "main.py", 'if __name__ == "__main__":\n    print("ok")\n')
    _save_file(server, session_id, "tests/test_ok.py", "def test_ok(): pass\n")

    client = _make_client(server)
    status, data = _post_json(
        server,
        client,
        "/api/generate",
        {"session_id": session_id, "prompt": "build a demo", "provider": "none"},
    )
    assert status == 200, data.decode("utf-8", errors="ignore")
    payload = json.loads(data)
    assert "commands" in payload
    commands = payload["commands"]
    assert any(c["cmd"] == "python main.py" for c in commands)
    assert any(c["cmd"] == "pytest" for c in commands)


def test_update_returns_command_list(server):
    """POST /api/update regenerates the workspace and returns runnable commands."""
    session_id = "test-update-commands"
    blueprint = (
        "metadata:\n"
        "  schema_version: \"3.0.0\"\n"
        "  project_name: demo\n"
        "  status: finalized\n"
        "  generation_method: llm_synthesized\n"
        "  llm_initialized: true\n"
        "manifest:\n"
        "  - path: main.py\n"
        "    lang: python\n"
        "    purpose: entrypoint\n"
        "execution_strategy:\n"
        "  primary_entrypoint: main.py\n"
        "  runtime: python3\n"
    )
    _save_file(server, session_id, "blueprint.aero", blueprint)

    client = _make_client(server)
    status, data = _post_json(
        server,
        client,
        "/api/update",
        {"session_id": session_id, "run_build": False, "force_overwrite": True},
    )
    assert status == 200, data.decode("utf-8", errors="ignore")
    payload = json.loads(data)
    assert "commands" in payload
    commands = payload["commands"]
    assert any(c["cmd"] == "python main.py" for c in commands)
