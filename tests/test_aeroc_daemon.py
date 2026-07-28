"""Python integration tests for the native aeroc-daemon executor."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from aero_forge._native import compile_aeroc, run_aeroc


def _make_spec(sources: list[tuple[str, bytes]]) -> dict:
    encoded = [
        {"path": p, "content_base64": base64.b64encode(c).decode("ascii")}
        for p, c in sources
    ]
    return {
        "nodes": ["lib", "app"],
        "edges": {"app": ["lib"]},
        "instructions": [
            {"op": "NOP"},
            {"op": "NOP"},
            {"op": "HALT"},
        ],
        "sources": encoded,
        "flags": 0,
    }


class TestAerocDaemon:
    """End-to-end checks for the mmap-backed wavefront executor."""

    def test_run_aeroc_nops(self, tmp_path: Path) -> None:
        """The daemon executes a trivial two-node DAG without errors."""
        output = tmp_path / "workspace.aeroc"
        spec = _make_spec([("src/lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }")])
        compile_aeroc(json.dumps(spec), str(output))
        run_aeroc(str(output), str(tmp_path), 2)

    def test_run_aeroc_content_hash_mismatch(self, tmp_path: Path) -> None:
        """A tampered .aeroc file is rejected before execution."""
        output = tmp_path / "workspace.aeroc"
        spec = _make_spec([("src/lib.rs", b"fn main() {}")])
        compile_aeroc(json.dumps(spec), str(output))
        raw = bytearray(output.read_bytes())
        # Flip a byte in the string table body.
        raw[200] ^= 0xFF
        output.write_bytes(raw)
        try:
            run_aeroc(str(output), str(tmp_path), 1)
        except ValueError as exc:
            assert "content hash mismatch" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for tampered container")
