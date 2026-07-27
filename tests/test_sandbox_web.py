"""Integration tests for the sandbox web terminal and blueprint endpoints."""

import json
import shutil
import socket
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from aero_forge.server import _resolve_port, make_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Start the web server on a free port with an isolated sandbox manager."""
    from aero_forge.sandbox.manager import SandboxManager
    from aero_forge.server import _manager as server_manager

    manager = SandboxManager(base_dir=tmp_path / "web-sessions")
    monkeypatch.setattr("aero_forge.server._manager", manager)

    port = _free_port()
    srv = make_server(port)
    http_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    http_thread.start()
    time.sleep(0.5)
    yield f"http://localhost:{port}"
    srv.shutdown()
    srv.server_close()


def _post_json(url: str, data: dict) -> tuple:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post_ndjson(url: str, data: dict) -> tuple:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        lines = [
            json.loads(line)
            for line in resp.read().decode("utf-8").splitlines()
            if line.strip()
        ]
    return resp.status, lines


def _get(url: str) -> tuple:
    try:
        with urlopen(url, timeout=10) as resp:
            return resp.status, resp.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def test_terminal_run_python_version(server):
    session_id = "test-terminal-python"
    command = "python3 --version"
    status, lines = _post_ndjson(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": command},
    )
    assert status == 200
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    output = "\n".join(
        line["data"] for line in lines if line.get("type") == "stdout"
    )
    assert "Python" in output


def test_terminal_run_cargo_version(server):
    if shutil.which("cargo") is None:
        pytest.skip("cargo not installed")

    session_id = "test-terminal-cargo"
    status, lines = _post_ndjson(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": "cargo --version"},
    )
    assert status == 200
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    output = "\n".join(
        line["data"] for line in lines if line.get("type") == "stdout"
    )
    assert "cargo" in output


def test_terminal_run_gpp_version(server):
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")

    session_id = "test-terminal-gpp"
    status, lines = _post_ndjson(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": "g++ --version"},
    )
    assert status == 200
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    output = "\n".join(
        line["data"] for line in lines if line.get("type") == "stdout"
    )
    assert "g++" in output or "Free Software Foundation" in output


def test_terminal_run_cwd_is_session_sandbox(server, tmp_path):
    from aero_forge.server import _manager

    session_id = "test-terminal-cwd"
    session_dir = _manager.create_session_sandbox(session_id)
    status, lines = _post_ndjson(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": "pwd"},
    )
    assert status == 200
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    output = "\n".join(
        line["data"] for line in lines if line.get("type") == "stdout"
    ).strip()
    assert output == str(session_dir.resolve())


def test_terminal_run_missing_command(server):
    session_id = "test-terminal-missing"
    status, body = _post_json(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": ""},
    )
    assert status == 400
    assert "Missing 'command'" in body["error"]


def test_api_blueprint_exposes_verification_nodes(server):
    session_id = "test-blueprint-nodes"
    blueprint = """
metadata:
  schema_version: "2.0.0"
  project_name: sandbox_test
  domain_target: pure_python
execution_strategy:
  primary_entrypoint:
    path: main.py
    runtime: python3
  cli_contract:
    parser_type: argparse
    flags: []
  run_spec: {}
abi_contracts: []
module_graph: []
verification_nodes:
  - test_id: cli_parses
    execution_cmd: python3 main.py --help
    expected_exit_code: 0
    stdout_match_patterns: [usage]
"""
    status, _ = _post_json(
        server + "/api/save-file",
        {"session_id": session_id, "path": "blueprint.aero", "content": blueprint},
    )
    assert status == 200

    status, body = _get(server + f"/api/blueprint?session_id={session_id}")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["session_id"] == session_id
    assert data["metadata"]["project_name"] == "sandbox_test"
    assert len(data["verification_nodes"]) == 1
    assert data["verification_nodes"][0]["test_id"] == "cli_parses"


def test_api_blueprint_missing_session(server):
    status, body = _get(server + "/api/blueprint")
    assert status == 400
    assert "Missing 'session_id'" in json.loads(body.decode("utf-8"))["error"]


def test_terminal_run_summary_includes_cwd_for_prompt(server, tmp_path):
    """The terminal run summary returns the sandbox cwd so the UI can render a prompt."""
    from aero_forge.server import _manager

    session_id = "test-terminal-prompt"
    session_dir = _manager.create_session_sandbox(session_id)
    status, lines = _post_ndjson(
        server + "/api/terminal/run",
        {"session_id": session_id, "command": "pwd"},
    )
    assert status == 200
    summary = lines[-1]
    assert summary["type"] == "summary"
    assert summary["exit_code"] == 0
    assert "cwd" in summary
    assert summary["cwd"] == str(session_dir.resolve())
