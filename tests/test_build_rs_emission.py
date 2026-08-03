"""Tests for valid build.rs emission across Rust emitters and materializers."""

from pathlib import Path

from aero_forge.builder.emitters.rust_emitter import RustEmitterPlugin
from aero_forge.builder.materializers.graph_materializer import GraphPolyglotMaterializer


def test_rust_emitter_plugin_emits_valid_build_rs() -> None:
    """RustEmitterPlugin always returns a build.rs containing fn main()."""
    plugin = RustEmitterPlugin()
    artifacts = plugin.emit_source_files(
        "rust_core",
        {"node_id": "rust_core", "lang": "rust", "source_files": ["src/lib.rs"]},
        [],
    )
    build_rs = next(a for a in artifacts if a.file_path.endswith("build.rs"))
    assert "fn main()" in build_rs.content
    assert "cargo:rerun-if-changed" in build_rs.content


def test_rust_emitter_cargo_manifest_references_build_rs() -> None:
    """Cargo.toml emitted by RustEmitterPlugin declares build = \"build.rs\"."""
    plugin = RustEmitterPlugin()
    manifest = plugin.emit_build_manifest("rust_core", ["pyo3"], [])
    assert manifest.file_path == "Cargo.toml"
    assert 'build = "build.rs"' in manifest.content


def test_graph_materializer_emits_valid_build_rs(tmp_path: Path) -> None:
    """GraphPolyglotMaterializer writes a build.rs with fn main for a Rust node."""
    workspace = tmp_path / "graph_workspace"
    spec = {
        "project": "demo",
        "architecture": "graph_polyglot",
        "nodes": [
            {
                "node_id": "rust_core",
                "lang": "rust",
                "toolchain": "cargo",
                "source_files": ["rust_core/src/lib.rs"],
                "exports": [],
            }
        ],
        "edges": [],
        "output_dir": str(tmp_path / "dist"),
    }
    materializer = GraphPolyglotMaterializer(workspace)
    result = materializer.materialize(spec)
    assert result["project"] == "demo"
    build_rs = workspace / "rust_core" / "build.rs"
    assert build_rs.exists()
    assert "fn main()" in build_rs.read_text(encoding="utf-8")
    cargo_toml = workspace / "rust_core" / "Cargo.toml"
    assert 'build = "build.rs"' in cargo_toml.read_text(encoding="utf-8")
