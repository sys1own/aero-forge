"""Embedded HTTP server for Aero-Forge web integration."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import mimetypes
import os
import pty
import re
import shutil
import struct
import termios
import threading
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import websockets

from aero_forge.chat import ChatSession
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge.sandbox.manager import SandboxManager

logger = logging.getLogger("aero_forge.server")

DEFAULT_PORT = 8080

_manager = SandboxManager()
_static_dir = Path(__file__).parent / "static"


def _session_dir(session_id: str) -> Path:
    return _manager.create_session_sandbox(session_id)


def _send_json(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
    handler.end_headers()
    handler.wfile.write(body)


def _send_bytes(
    handler: BaseHTTPRequestHandler,
    status: int,
    data: bytes,
    content_type: str,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
    if headers:
        for key, value in headers.items():
            handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(data)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return b""
    return handler.rfile.read(length)


def _parse_json_body(handler: BaseHTTPRequestHandler) -> Any:
    raw = _read_body(handler)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _parse_multipart(body: bytes, boundary: bytes) -> Optional[bytes]:
    """Return the first file payload found in a multipart/form-data body."""
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers = part[:header_end].decode("utf-8", errors="ignore")
        if "filename=" in headers or "Content-Type:" in headers:
            return part[header_end + 4 :]
    return None


def _extract_zip_safely(zip_bytes: bytes, dest: Path) -> None:
    """Extract a zip archive to ``dest`` while guarding against path traversal."""
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            try:
                target.relative_to(dest.resolve())
            except ValueError as exc:
                raise ValueError(f"Zip member escapes extraction directory: {member}") from exc
        zf.extractall(dest)


def _build_tree(directory: Path, rel: Optional[Path] = None) -> Dict[str, Any]:
    """Return a nested JSON tree of files and directories."""
    rel = rel or Path(".")
    name = "." if rel == Path(".") else (rel.name or directory.name or ".")
    node: Dict[str, Any] = {
        "name": name,
        "type": "directory",
        "path": str(rel),
        "children": [],
    }
    for path in sorted(directory.iterdir()):
        child_rel = rel / path.name
        if path.is_dir():
            if path.name in {"__pycache__", ".git", ".variant_0", ".variant_1", ".variant_2"}:
                continue
            node["children"].append(_build_tree(path, child_rel))
        else:
            node["children"].append(
                {
                    "name": path.name,
                    "type": "file",
                    "path": str(child_rel),
                    "size": path.stat().st_size,
                }
            )
    return node


def _resolve_file(session_dir: Path, file_path: str) -> Path:
    """Resolve ``file_path`` under ``session_dir`` and guard against traversal."""
    target = (session_dir / file_path).resolve()
    target.relative_to(session_dir.resolve())
    return target


class AeroForgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Aero-Forge web API."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/api/files":
                return self._handle_files(query)
            if path == "/api/file-content":
                return self._handle_file_content(query)
            if path == "/api/download-zip":
                return self._handle_download_zip(query)

            return self._serve_static(path)
        except Exception as exc:
            logger.exception("GET %s failed", self.path)
            return _send_json(self, 500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/build":
                return self._handle_build()
            if path == "/api/chat":
                return self._handle_chat()
            if path == "/api/upload-zip":
                return self._handle_upload_zip()
            if path == "/api/save-file":
                return self._handle_save_file()
            if path == "/api/create-node":
                return self._handle_create_node()
            if path == "/api/rename-node":
                return self._handle_rename_node()
            if path == "/api/delete-node":
                return self._handle_delete_node()

            return _send_json(self, 404, {"error": "Not found"})
        except Exception as exc:
            logger.exception("POST %s failed", self.path)
            return _send_json(self, 500, {"error": str(exc)})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key, X-API-Key, Authorization")
        self.end_headers()

    def _api_key(self, body: Dict[str, Any]) -> Optional[str]:
        """Resolve API key from JSON body, X-Api-Key header, or Authorization header."""
        key = body.get("api_key")
        if key:
            return key
        key = self.headers.get("X-Api-Key") or self.headers.get("X-API-Key")
        if key:
            return key
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1]
        return None

    def _handle_build(self) -> None:
        try:
            body = _parse_json_body(self)
            prompt = body.get("prompt", "").strip()
            if not prompt:
                return _send_json(self, 400, {"error": "Missing 'prompt'"})

            session_id = body.get("session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)

            variants = 3 if body.get("variants") else 1
            config = ConfigOverride(
                llm_provider=body.get("provider"),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )

            result = generate_and_build(
                prompt,
                output_dir=session_dir,
                project_name="generated",
                llm_provider=config.llm_provider,
                model=config.model,
                max_retries=3,
                max_iterations=5,
                variants=variants,
                build_kwargs={"max_workers": 1, "cache_enabled": False},
                config_override=config,
            )

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "success" if result.get("build", {}).get("success") else "failure",
                    "result": result,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Build endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_save_file(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            content = body.get("content")
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})
            if content is None:
                return _send_json(self, 400, {"error": "Missing 'content'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "status": "saved",
                    "size": target.stat().st_size,
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Save-file endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_chat(self) -> None:
        try:
            body = _parse_json_body(self)
            message = body.get("message", "").strip()
            if not message:
                return _send_json(self, 400, {"error": "Missing 'message'"})

            session_id = body.get("session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)
            history = body.get("history", [])

            config = ConfigOverride(
                llm_provider=body.get("provider"),
                api_key=self._api_key(body),
                model=body.get("model"),
                max_retries=3,
            )

            chat = ChatSession(
                session_dir,
                llm_provider=config.llm_provider,
                model=config.model,
                api_key=config.api_key,
                session_id=session_id,
                max_retries=3,
                config_override=config,
            )
            if history:
                chat.messages = history

            response = chat.process(message)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "response": response,
                    "messages": chat.messages,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Chat endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_files(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        if not session_id:
            return _send_json(self, 400, {"error": "Missing 'session_id'"})

        session_dir = _manager._session_dir(session_id)
        if not session_dir.is_dir():
            return _send_json(self, 404, {"error": f"Session {session_id} not found"})

        return _send_json(
            self,
            200,
            {
                "session_id": session_id,
                "tree": _build_tree(session_dir),
            },
        )

    def _handle_file_content(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        file_path = _first(query, "path")
        if not session_id or not file_path:
            return _send_json(
                self, 400, {"error": "Missing 'session_id' and/or 'path'"}
            )

        session_dir = _manager._session_dir(session_id)
        try:
            target = _resolve_file(session_dir, file_path)
        except ValueError:
            return _send_json(self, 400, {"error": "Invalid path"})

        if not target.is_file():
            return _send_json(self, 404, {"error": "File not found"})

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _send_json(self, 400, {"error": "File is not text-readable"})

        return _send_json(
            self,
            200,
            {
                "session_id": session_id,
                "path": file_path,
                "content": content,
            },
        )

    def _handle_upload_zip(self) -> None:
        try:
            body = _read_body(self)
            if not body:
                return _send_json(self, 400, {"error": "Empty body"})

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                match = re.search(r'boundary=([^;\s]+)', content_type)
                if not match:
                    return _send_json(self, 400, {"error": "Missing multipart boundary"})
                boundary = match.group(1).encode("utf-8")
                zip_bytes = _parse_multipart(body, boundary)
                if zip_bytes is None:
                    return _send_json(self, 400, {"error": "No file found in multipart body"})
            else:
                zip_bytes = body

            # Read session id from a query parameter if present, otherwise generate one.
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            session_id = _first(query, "session_id") or str(uuid.uuid4())
            session_dir = _session_dir(session_id)

            _extract_zip_safely(zip_bytes, session_dir)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "status": "uploaded",
                    "files": _build_tree(session_dir),
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("Upload endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_create_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            is_dir = bool(body.get("is_dir", False))
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if target.exists():
                return _send_json(self, 409, {"error": "Node already exists"})

            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "is_dir": is_dir,
                    "status": "created",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Create-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_rename_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            old_path = body.get("old_path", "").strip()
            new_path = body.get("new_path", "").strip()
            if not session_id or not old_path or not new_path:
                return _send_json(self, 400, {"error": "Missing 'session_id', 'old_path', and/or 'new_path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                source = _resolve_file(session_dir, old_path)
                target = _resolve_file(session_dir, new_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if not source.exists():
                return _send_json(self, 404, {"error": "Source not found"})
            if target.exists():
                return _send_json(self, 409, {"error": "Target already exists"})

            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "old_path": old_path,
                    "new_path": new_path,
                    "status": "renamed",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Rename-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_delete_node(self) -> None:
        try:
            body = _parse_json_body(self)
            session_id = body.get("session_id", "").strip()
            file_path = body.get("path", "").strip()
            if not session_id or not file_path:
                return _send_json(self, 400, {"error": "Missing 'session_id' and/or 'path'"})

            session_dir = _manager.create_session_sandbox(session_id)
            try:
                target = _resolve_file(session_dir, file_path)
            except ValueError:
                return _send_json(self, 400, {"error": "Invalid path"})

            if not target.exists():
                return _send_json(self, 404, {"error": "Node not found"})

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            return _send_json(
                self,
                200,
                {
                    "session_id": session_id,
                    "path": file_path,
                    "status": "deleted",
                },
            )
        except ValueError as exc:
            return _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Delete-node endpoint failed")
            return _send_json(self, 500, {"error": str(exc)})

    def _handle_download_zip(self, query: Dict[str, List[str]]) -> None:
        session_id = _first(query, "session_id")
        if not session_id:
            return _send_json(self, 400, {"error": "Missing 'session_id'"})

        try:
            archive_bytes = _manager.archive_session_sandbox(session_id)
        except ValueError as exc:
            return _send_json(self, 404, {"error": str(exc)})

        return _send_bytes(
            self,
            200,
            archive_bytes,
            "application/zip",
            {
                "Content-Disposition": f'attachment; filename="{session_id}.zip"'
            },
        )

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        local_path = _static_dir / path.lstrip("/")
        if not local_path.is_file() or not str(local_path.resolve()).startswith(
            str(_static_dir.resolve())
        ):
            return _send_json(self, 404, {"error": "Not found"})

        content_type, _ = mimetypes.guess_type(str(local_path))
        content_type = content_type or "application/octet-stream"
        data = local_path.read_bytes()
        return _send_bytes(self, 200, data, content_type)


def _first(query: Dict[str, List[str]], key: str) -> Optional[str]:
    values = query.get(key)
    return values[0] if values else None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that allows immediate rebinding to the same port."""

    allow_reuse_address = True


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Return a threaded HTTP server bound to the given port."""
    return ReusableThreadingHTTPServer(("", port), AeroForgeHandler)


def _set_pty_size(master_fd: int, cols: int, rows: int) -> None:
    """Set the PTY window size so tools like top, htop and editors render."""
    try:
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError as exc:
        logger.debug("Could not set PTY size: %s", exc)


def _child_setup(master_fd: int, slave_fd: int) -> None:
    """Close the PTY master fd in the child before exec'ing the shell."""
    try:
        os.close(master_fd)
    except OSError:
        pass


async def _handle_terminal(websocket: Any) -> None:
    """Spawn a per-session subshell over a PTY and relay I/O to the WebSocket."""
    request_path = websocket.request.path if websocket.request else "/ws/terminal"
    parsed = urlparse(request_path)
    if parsed.path != "/ws/terminal":
        await websocket.close(code=1002, reason="Invalid path")
        return

    query = parse_qs(parsed.query)
    session_id = _first(query, "session_id") or str(uuid.uuid4())
    session_dir = _manager.create_session_sandbox(session_id)

    shell = shutil.which("bash") or shutil.which("sh")
    if not shell:
        await websocket.close(code=1011, reason="No shell available on this system")
        return

    master_fd: int = -1
    process: Optional[Any] = None
    reader_added = False
    loop = asyncio.get_running_loop()

    try:
        master_fd, slave_fd = pty.openpty()
        os.set_blocking(master_fd, False)

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["AERO_FORGE_SESSION"] = session_id
        env["AERO_FORGE_SESSION_DIR"] = str(session_dir)

        process = await asyncio.create_subprocess_exec(
            shell,
            "-i",
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=session_dir,
            env=env,
            start_new_session=True,
            preexec_fn=lambda: _child_setup(master_fd, slave_fd),
        )
        try:
            os.close(slave_fd)
        except OSError:
            pass

        _set_pty_size(master_fd, 80, 24)

        async def _send_to_client(data: bytes) -> None:
            try:
                await websocket.send(data)
            except Exception:
                pass

        def _on_master_readable() -> None:
            try:
                data = os.read(master_fd, 4096)
            except (BlockingIOError, OSError):
                return
            if data:
                asyncio.create_task(_send_to_client(data))
            else:
                loop.remove_reader(master_fd)

        loop.add_reader(master_fd, _on_master_readable)
        reader_added = True

        async def _wait_for_process() -> None:
            if process is None:
                return
            try:
                await process.wait()
            except Exception:
                pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        asyncio.create_task(_wait_for_process())

        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            if not data:
                continue

            # Resize messages are sent by xterm.js as JSON {cols, rows}
            if data.startswith(b"{"):
                try:
                    payload = json.loads(data.decode("utf-8"))
                    if isinstance(payload, dict) and "cols" in payload and "rows" in payload:
                        _set_pty_size(master_fd, int(payload["cols"]), int(payload["rows"]))
                        continue
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            try:
                os.write(master_fd, data)
            except (BlockingIOError, OSError):
                break

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if reader_added and master_fd >= 0:
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2)
            except Exception:
                pass
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass


async def _run_websocket_server(ws_port: int) -> None:
    """Run the terminal WebSocket server on the given port."""
    async with websockets.serve(_handle_terminal, "", ws_port, ping_interval=20, ping_timeout=10):
        logger.info("Terminal WebSocket server listening on ws://localhost:%s/ws/terminal", ws_port)
        await asyncio.Future()  # run forever


def _start_websocket_server(ws_port: int) -> None:
    """Start the asyncio WebSocket server in a dedicated daemon thread."""
    asyncio.run(_run_websocket_server(ws_port))


def run_server(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the web server and optionally open the user's browser."""
    _static_dir.mkdir(parents=True, exist_ok=True)
    server = make_server(port)

    def http_serve() -> None:
        try:
            server.serve_forever()
        finally:
            server.server_close()

    http_thread = threading.Thread(target=http_serve, daemon=True)
    http_thread.start()

    ws_port = port + 1
    ws_thread = threading.Thread(target=_start_websocket_server, args=(ws_port,), daemon=True)
    ws_thread.start()

    url = f"http://localhost:{port}"
    logger.info("Aero-Forge web server running at %s", url)
    logger.info("Terminal WebSocket available at ws://localhost:%s/ws/terminal", ws_port)

    if open_browser:
        import webbrowser

        webbrowser.open(url)

    try:
        while http_thread.is_alive():
            http_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("Shutting down web server...")
    finally:
        server.shutdown()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_server()
