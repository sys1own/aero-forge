"""Tests for .aeroc/zip blueprint ingestion and safe materialization guards."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from aero_forge.blueprint.blueprint_parser import is_blueprint_ready, load_blueprint
from aero_forge.builder.aeroc_compiler import compile_directory_to_aeroc
from aero_forge.scaffold.workspace import BlueprintRegenerator


def _post_json(base_url, client, path, payload):
    body = json.dumps(payload).encode("utf-8")
    client.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = client.getresponse()
    data = resp.read()
    return resp.status, data


def _post_bytes(base_url, client, path, data, query=""):
    full_path = f"{path}?{query}" if query else path
    client.request("POST", full_path, body=data, headers={"Content-Type": "application/octet-stream"})
    resp = client.getresponse()
    return resp.status, resp.read()


def _make_client(base_url):
    from http.client import HTTPConnection
    return HTTPConnection(base_url.replace("http://", "").replace("https://", ""))


@pytest.fixture
def server(tmp_path, monkeypatch):
    from aero_forge.sandbox.manager import SandboxManager
    from aero_forge.server import _manager as manager_global
    from aero_forge.server import make_server
    import threading

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


def _free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _save_file(base_url, session_id, path, content):
    client = _make_client(base_url)
    _post_json(base_url, client, "/api/save-file", {"session_id": session_id, "path": path, "content": content})


class TestBlueprintReadiness:
    def test_load_blueprint_reads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "blueprint.aero"
        path.write_text("metadata:\n  status: finalized\n")
        data = load_blueprint(path)
        assert data == {"metadata": {"status": "finalized"}}

    def test_is_blueprint_ready_returns_false_for_draft(self) -> None:
        blueprint = {
            "metadata": {"status": "draft"},
            "manifest": [{"path": "main.py", "lang": "python"}],
        }
        assert is_blueprint_ready(blueprint) is False

    def test_is_blueprint_ready_returns_false_without_code_nodes(self) -> None:
        blueprint = {"metadata": {"status": "finalized"}}
        assert is_blueprint_ready(blueprint) is False

    def test_is_blueprint_ready_returns_true_for_llm_finalized(self) -> None:
        blueprint = {
            "metadata": {
                "status": "finalized",
                "generation_method": "llm_synthesized",
                "llm_initialized": True,
            },
            "contracts": [{"name": "add", "signature": "def add(a: float, b: float) -> float"}],
        }
        assert is_blueprint_ready(blueprint) is True

    def test_is_blueprint_ready_accepts_path(self, tmp_path: Path) -> None:
        path = tmp_path / "workspace_blueprint.yaml"
        path.write_text(
            "metadata:\n"
            "  status: finalized\n"
            "  llm_initialized: true\n"
            "contracts:\n"
            "  - name: add\n"
        )
        assert is_blueprint_ready(path) is True


class TestAerocIngestion:
    def test_aeroc_with_source_suppresses_uninitialized_prompt(self, server, tmp_path: Path) -> None:
        """An .aeroc that already contains source files is loaded directly."""
        workspace = tmp_path / "aeroc_src"
        workspace.mkdir()
        (workspace / "main.py").write_text('print("hello")\n')
        aeroc = tmp_path / "workspace.aeroc"
        compile_directory_to_aeroc(workspace, aeroc)

        session_id = "test-aeroc-source"
        client = _make_client(server)
        status, data = _post_bytes(
            server,
            client,
            "/api/upload-aeroc",
            aeroc.read_bytes(),
            query=f"session_id={session_id}&target_path=.",
        )
        assert status == 200, data.decode("utf-8", errors="ignore")
        payload = json.loads(data)
        assert payload["status"] == "success"
        assert payload.get("blueprint_source") == "aeroc_archive"
        assert payload.get("auto_initialized") is True
        commands = payload["commands"]
        assert any(c["cmd"] == "python main.py" for c in commands)


class TestSafeMaterialization:
    def test_materializer_aborts_uninitialized_blueprint(self, tmp_path: Path) -> None:
        """Regenerating from a draft blueprint on a non-empty workspace must not delete code."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        source = workspace / "main.py"
        source.write_text("print('keep me')\n")
        blueprint = workspace / "blueprint.aero"
        blueprint.write_text(
            "metadata:\n"
            "  status: draft\n"
            "manifest:\n"
            "  - path: main.py\n"
            "    lang: python\n"
        )

        regenerator = BlueprintRegenerator(workspace)
        with pytest.raises(ValueError, match="Cannot materialize: Blueprint is uninitialized"):
            regenerator.run()

        # Ensure the existing source file was not touched.
        assert source.read_text() == "print('keep me')\n"

    def test_materializer_requires_force_for_non_empty_workspace(self, tmp_path: Path) -> None:
        """A non-empty workspace needs force_overwrite to regenerate."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "existing.py").write_text("pass\n")
        blueprint = workspace / "blueprint.aero"
        blueprint.write_text(
            "metadata:\n"
            "  status: finalized\n"
            "  generation_method: llm_synthesized\n"
            "manifest:\n"
            "  - path: main.py\n"
            "    lang: python\n"
            "contracts:\n"
            "  - name: add\n"
            "    signature: \"def add(a: float, b: float) -> float:\"\n"
        )

        regenerator = BlueprintRegenerator(workspace, force_overwrite=False)
        with pytest.raises(ValueError, match="Workspace is not empty"):
            regenerator.run()

        # With force_overwrite the regeneration proceeds (and creates a backup).
        regenerator = BlueprintRegenerator(workspace, force_overwrite=True, keep_backup=True)
        result = regenerator.run()
        assert result["status"] == "success"
        assert (workspace / ".aero_backup").is_dir()
