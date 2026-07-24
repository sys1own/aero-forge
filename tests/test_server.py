"""Integration tests for the embedded Aero-Forge web server."""

import asyncio
import io
import json
import socket
import threading
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import websockets

from aero_forge.sandbox.manager import SandboxManager
from aero_forge.server import make_server


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
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    # Give the aiohttp server a moment to bind before tests connect.
    import time
    time.sleep(0.5)
    yield f"http://localhost:{port}"
    server.shutdown()
    server.server_close()
    http_thread.join(timeout=2)


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
    assert (b"Aero Forge" in body or b"Aero-Forge" in body)
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


def test_api_save_file(server):
    session_id = "test-session-save"
    zip_bytes = _make_zip({"src/main.py": "def add(a, b):\n    return a + b\n"})
    status, _ = _post_bytes(
        server + f"/api/upload-zip?session_id={session_id}", zip_bytes
    )
    assert status == 200

    status, body = _post_json(
        server + "/api/save-file",
        {"session_id": session_id, "path": "src/main.py", "content": "def add(a, b):\n    return a + b + 1\n"},
    )
    assert status == 200
    assert body["status"] == "saved"
    assert body["path"] == "src/main.py"

    status, body = _get(server + f"/api/file-content?session_id={session_id}&path=src/main.py")
    assert status == 200
    file_data = json.loads(body.decode("utf-8"))
    assert "return a + b + 1" in file_data["content"]

    try:
        status, body = _post_json(
            server + "/api/save-file",
            {"session_id": session_id, "path": "../etc/passwd", "content": "hacked"},
        )
    except HTTPError as exc:
        status, body = exc.code, exc.read()
    assert status == 400


def test_api_create_rename_delete_node(server):
    session_id = "test-session-crud"

    status, body = _post_json(
        server + "/api/create-node",
        {"session_id": session_id, "path": "src/new.py", "is_dir": False},
    )
    assert status == 200
    assert body["status"] == "created"

    status, body = _post_json(
        server + "/api/create-node",
        {"session_id": session_id, "path": "lib", "is_dir": True},
    )
    assert status == 200
    assert body["is_dir"] is True

    status, body = _post_json(
        server + "/api/rename-node",
        {"session_id": session_id, "old_path": "src/new.py", "new_path": "src/renamed.py"},
    )
    assert status == 200
    assert body["status"] == "renamed"

    status, body = _get(server + f"/api/files?session_id={session_id}")
    assert status == 200
    tree = json.loads(body.decode("utf-8"))
    paths = _collect_paths(tree["tree"])
    assert "src/renamed.py" in paths
    assert "src/new.py" not in paths
    assert _node_exists(tree["tree"], "lib", "directory")

    status, body = _post_json(
        server + "/api/delete-node",
        {"session_id": session_id, "path": "lib"},
    )
    assert status == 200
    assert body["status"] == "deleted"

    status, body = _get(server + f"/api/files?session_id={session_id}")
    assert status == 200
    tree = json.loads(body.decode("utf-8"))
    assert not _node_exists(tree["tree"], "lib", "directory")


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


def test_api_run_streaming(server):
    session_id = "test-session-run"
    status, _ = _post_json(
        server + "/api/create-node",
        {"session_id": session_id, "path": "main.py", "is_dir": False},
    )
    assert status == 200

    status, _ = _post_json(
        server + "/api/save-file",
        {"session_id": session_id, "path": "main.py", "content": "print('aero_forge_run_ok')"},
    )
    assert status == 200

    body_json = json.dumps(
        {"session_id": session_id, "path": "main.py"}
    ).encode("utf-8")
    req = Request(server + "/api/run", data=body_json, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        raw = resp.read().decode("utf-8")

    lines = [line for line in raw.strip().split("\n") if line]
    assert any("aero_forge_run_ok" in line for line in lines)
    summary = json.loads(lines[-1])
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    assert summary["file"] == "main.py"


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


def test_api_files_creates_session_lazily(server):
    status, body = _get(server + "/api/files?session_id=does-not-exist-yet")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["session_id"] == "does-not-exist-yet"
    assert data["tree"]["type"] == "directory"


def test_websocket_terminal(server):
    port = int(server.rsplit(":", 1)[-1])
    ws_url = f"ws://localhost:{port}/ws/terminal?session_id=test-ws-terminal"

    async def _client():
        async with websockets.connect(ws_url) as ws:
            # Wait for shell prompt / motd
            await asyncio.sleep(0.5)
            await ws.send("echo aero_forge_terminal_test\r")
            output = b""
            for _ in range(30):
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    break
                if isinstance(data, str):
                    data = data.encode("utf-8")
                output += data
                if b"aero_forge_terminal_test" in output:
                    break
            return output

    output = asyncio.run(_client())
    assert b"aero_forge_terminal_test" in output


def _node_exists(node, name, node_type):
    if node["name"] == name and node["type"] == node_type:
        return True
    for child in node.get("children", []):
        if _node_exists(child, name, node_type):
            return True
    return False


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
