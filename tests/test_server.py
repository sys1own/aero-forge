"""Integration tests for the embedded Aero-Forge web server."""

import io
import json
import socket
import threading
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from aero_forge.sandbox.manager import SandboxManager
from aero_forge.server import AeroForgeHandler, make_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Start the web server on a free port with an isolated sandbox manager."""
    manager = SandboxManager(base_dir=tmp_path / "web-sessions")
    monkeypatch.setattr("aero_forge.server._manager", manager)

    port = _free_port()
    server = make_server(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _get(url: str) -> tuple:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _post_json(url: str, data: dict) -> tuple:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post_bytes(url: str, data: bytes, content_type: str = "application/zip") -> tuple:
    req = Request(
        url,
        data=data,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _make_zip(file_map: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in file_map.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_index_html(server):
    status, body = _get(server + "/")
    assert status == 200
    assert b"Aero-Forge" in body
    assert b"</html>" in body


def test_static_404(server):
    status, body = _get(server + "/does-not-exist.txt")
    assert status == 404
    assert b"Not found" in body


def test_api_upload_zip_and_files(server):
    session_id = "test-session-files"
    zip_bytes = _make_zip({"src/main.py": "def add(a, b):\n    return a + b\n"})
    status, body = _post_bytes(
        server + f"/api/upload-zip?session_id={session_id}", zip_bytes
    )
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["session_id"] == session_id
    assert data["status"] == "uploaded"

    status, body = _get(server + f"/api/files?session_id={session_id}")
    assert status == 200
    tree = json.loads(body.decode("utf-8"))
    assert tree["session_id"] == session_id
    paths = _collect_paths(tree["tree"])
    assert "src/main.py" in paths

    status, body = _get(
        server + f"/api/file-content?session_id={session_id}&path=src/main.py"
    )
    assert status == 200
    file_data = json.loads(body.decode("utf-8"))
    assert file_data["path"] == "src/main.py"
    assert "def add" in file_data["content"]


def test_api_download_zip(server):
    session_id = "test-session-download"
    zip_bytes = _make_zip({"data.txt": "hello"})
    status, _ = _post_bytes(
        server + f"/api/upload-zip?session_id={session_id}", zip_bytes
    )
    assert status == 200

    status, body = _get(server + f"/api/download-zip?session_id={session_id}")
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(body), "r") as zf:
        assert "data.txt" in zf.namelist()
        assert zf.read("data.txt").decode("utf-8") == "hello"


def test_api_build_passes_config_override(server, monkeypatch):
    captured = {}

    def fake_generate_and_build(prompt, *, config_override=None, variants=1, **kwargs):
        captured["prompt"] = prompt
        captured["config_override"] = config_override
        captured["variants"] = variants
        captured["kwargs"] = kwargs
        return {
            "source_path": "src/generated.py",
            "test_path": "tests/test_generated.py",
            "build": {"success": True, "passed": 1, "total": 1},
        }

    monkeypatch.setattr("aero_forge.server.generate_and_build", fake_generate_and_build)

    status, data = _post_json(
        server + "/api/build",
        {
            "prompt": "Build a fast fibonacci function",
            "provider": "deepseek",
            "api_key": "sk-test",
            "variants": True,
        },
    )
    assert status == 200
    assert data["status"] == "success"
    assert "session_id" in data
    assert captured["prompt"] == "Build a fast fibonacci function"
    assert captured["variants"] == 3
    assert captured["config_override"] is not None
    assert captured["config_override"].llm_provider == "deepseek"
    assert captured["config_override"].api_key == "sk-test"


def test_api_chat_passes_config_override(server, monkeypatch):
    captured = {}

    class FakeChatSession:
        def __init__(self, output_dir, *, config_override=None, **kwargs):
            captured["config_override"] = config_override
            self.messages = []

        def process(self, text):
            return f"Echo: {text}"

    monkeypatch.setattr("aero_forge.server.ChatSession", FakeChatSession)

    status, data = _post_json(
        server + "/api/chat",
        {
            "message": "hello",
            "provider": "openai",
            "api_key": "sk-chat",
        },
    )
    assert status == 200
    assert data["response"] == "Echo: hello"
    assert captured["config_override"] is not None
    assert captured["config_override"].llm_provider == "openai"
    assert captured["config_override"].api_key == "sk-chat"


def test_api_file_content_path_traversal(server):
    session_id = "test-session-traversal"
    zip_bytes = _make_zip({"src/main.py": "x = 1\n"})
    _post_bytes(server + f"/api/upload-zip?session_id={session_id}", zip_bytes)

    status, body = _get(
        server + f"/api/file-content?session_id={session_id}&path=../outside.py"
    )
    assert status == 400
    data = json.loads(body.decode("utf-8"))
    assert "Invalid path" in data["error"]


def test_api_files_missing_session(server):
    status, body = _get(server + "/api/files?session_id=does-not-exist")
    assert status == 404
    data = json.loads(body.decode("utf-8"))
    assert "not found" in data["error"].lower()


def _collect_paths(node, prefix=""):
    paths = []
    if node["type"] == "file":
        return [f"{prefix}/{node['name']}".lstrip("/")]
    dir_prefix = prefix
    if node["name"] != ".":
        dir_prefix = f"{prefix}/{node['name']}".lstrip("/")
    for child in node.get("children", []):
        paths.extend(_collect_paths(child, dir_prefix))
    return paths
