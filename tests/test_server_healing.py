"""Tests for self-healing endpoints."""

import json
import socket
import threading

import pytest

from aero_forge.sandbox.manager import SandboxManager
from aero_forge.server import _resolve_port, make_server


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
    import time

    time.sleep(0.5)
    yield f"http://localhost:{port}"
    server.shutdown()
    server.server_close()
    http_thread.join(timeout=2)


def _post_json(url: str, payload: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_evaluate_error_detects_python_name_error(server):
    result = _post_json(
        server + "/api/workspace/evaluate-error",
        {
            "session_id": "test-heal-eval",
            "command": "python main.py",
            "exit_code": 1,
            "log_text": "Traceback\n  File 'main.py', line 3, in <module>\n    print(math.pi)\nNameError: name 'math' is not defined",
        },
    )
    assert result["healable"] is True
    assert result["error_type"] == "python_name_error"


def test_heal_endpoint_patches_missing_import(server, tmp_path):
    """Create a session with a broken main.py, then heal it."""
    session_id = "test-heal-session"
    _post_json(
        server + "/api/save-file",
        {
            "session_id": session_id,
            "path": "main.py",
            "content": "print(math.sqrt(16))\n",
        },
    )
    log_text = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 1, in <module>\n'
        "    print(math.sqrt(16))\n"
        "NameError: name 'math' is not defined\n"
    )
    result = _post_json(
        server + "/api/workspace/heal",
        {
            "session_id": session_id,
            "command": "python main.py",
            "exit_code": 1,
            "log_text": log_text,
        },
    )
    assert result["status"] == "success"
    assert result.get("strategy_used") == "ast"
    assert result.get("target_file") == "main.py"
    assert "main.py" in result["patched_files"]
    assert "import math" in result["diff"]


def test_chat_fix_error_triggers_healing(server):
    """POST /api/chat with a terminal error context and 'fix error' triggers a patch."""
    session_id = "test-chat-heal"
    _post_json(
        server + "/api/save-file",
        {
            "session_id": session_id,
            "path": "main.py",
            "content": "print(math.sqrt(16))\n",
        },
    )
    result = _post_json(
        server + "/api/chat",
        {
            "session_id": session_id,
            "message": "fix error",
            "terminal_command": "python main.py",
            "terminal_exit_code": 1,
            "terminal_log_text": "NameError: name 'math' is not defined",
        },
    )
    assert result["session_id"] == session_id
    # The reply should mention the applied patch and re-running.
    assert "patch" in result["reply"].lower() or "re-run" in result["reply"].lower()
