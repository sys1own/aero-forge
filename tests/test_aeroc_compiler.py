"""Python-side integration tests for the workspace.aeroc compiler."""

import base64
import json
import os
import struct
from pathlib import Path

import pytest

from aero_forge.blueprint.schema import (
    ArtifactType,
    BlueprintV3,
    BuildArtifact,
    ExecutionStrategyV3,
    Metadata,
)
from aero_forge.builder.aeroc_compiler import compile_blueprint_to_aeroc
from aero_forge._native import compile_aeroc


def _make_spec(sources: list) -> dict:
    """Return a minimal aeroc project spec JSON dict."""
    encoded = [
        {"path": p, "content_base64": base64.b64encode(c).decode("ascii")}
        for p, c in sources
    ]
    return {
        "nodes": ["lib", "app"],
        "edges": {"app": ["lib"]},
        "instructions": [
            {
                "op": "CARGO_BUILD",
                "manifest_ref": "Cargo.toml",
                "flags": 0x01,
            },
            {
                "op": "PYO3_BIND",
                "src_ref": "src/lib.rs",
                "out_ref": "dist/pymodule.so",
            },
            {"op": "HALT"},
        ],
        "sources": encoded,
        "flags": 0,
    }


class TestAerocCompiler:
    """End-to-end checks for the .aeroc binary container."""

    def test_compile_aeroc_creates_binary(self, tmp_path: Path) -> None:
        """compile_aeroc writes a file containing the expected header."""
        output = tmp_path / "workspace.aeroc"
        spec = _make_spec([("src/main.rs", b"fn main() {}")])
        content_hash = compile_aeroc(json.dumps(spec), str(output))

        assert output.is_file()
        assert len(content_hash) == 32  # hex-encoded 16 bytes

        raw = output.read_bytes()
        assert len(raw) >= 128
        assert raw[:8] == b"AEROFOG\0"
        major, minor = struct.unpack_from("<HH", raw, 8)
        assert major == 1
        assert minor == 0
        header_size = struct.unpack_from("<I", raw, 16)[0]
        assert header_size == 128

    def test_compile_aeroc_layout(self, tmp_path: Path) -> None:
        """Sections are placed in the expected order and the payload is reachable."""
        output = tmp_path / "workspace.aeroc"
        source = b"fn main() {}\n" * 1024  # large enough to trigger chunking?
        spec = _make_spec([("src/main.rs", source)])
        compile_aeroc(json.dumps(spec), str(output))

        raw = output.read_bytes()
        (
            st_off,
            st_len,
            dag_off,
            node_count,
            dag_stride,
            bc_off,
            bc_len,
            pl_off,
            pl_len,
            dict_off,
            dict_len,
        ) = struct.unpack_from("<QQQIIQQQQQI", raw, 20)

        assert st_off == 128
        assert dag_off == st_off + st_len
        assert bc_off == dag_off + (dag_stride * node_count)
        assert pl_off == bc_off + bc_len
        assert dict_off == pl_off + pl_len
        assert dict_len <= 64 * 1024

    def test_compile_aeroc_deterministic(self, tmp_path: Path) -> None:
        """Two compilations of the same spec yield identical bytes."""
        output1 = tmp_path / "one.aeroc"
        output2 = tmp_path / "two.aeroc"
        spec = _make_spec([("src/lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }\n")])
        h1 = compile_aeroc(json.dumps(spec), str(output1))
        h2 = compile_aeroc(json.dumps(spec), str(output2))

        assert h1 == h2
        assert output1.read_bytes() == output2.read_bytes()


def test_compile_blueprint_to_aeroc(tmp_path: Path) -> None:
    """A Blueprint v3 can be converted into a .aeroc binary container."""
    src = tmp_path / "src" / "main.rs"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fn main() {}")

    blueprint = BlueprintV3(
        metadata=Metadata(project_name="demo"),
        build_pipeline=[
            BuildArtifact(
                id="lib",
                type=ArtifactType.cargo_cdylib,
                source_files=["src/main.rs"],
                output_path="dist/libdemo.so",
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint="src/main.rs", runtime="cargo"
        ),
    )
    output = tmp_path / "workspace.aeroc"
    content_hash = compile_blueprint_to_aeroc(blueprint, output, workspace=tmp_path)

    assert output.is_file()
    assert len(content_hash) == 32
    raw = output.read_bytes()
    assert raw[:8] == b"AEROFOG\0"
