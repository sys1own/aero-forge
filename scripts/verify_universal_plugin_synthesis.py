#!/usr/bin/env python3
"""Live verification of Universal Plugin Synthesis for Aero-Forge.

Exercises JIT emitter synthesis for Zig, Mojo, and Go, confirms that the
SystemToolchainRouter can invoke the synthesized toolchains, and runs the
full Zig math-kernel build end-to-end via GraphPolyglotMaterializer.

Usage:
    DEEPSEEK_API_KEY=... python scripts/verify_universal_plugin_synthesis.py
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)
from aero_forge.builder.language_router import SystemToolchainRouter
from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)
from aero_forge.builder.intent_compiler import IntentCompiler
from aero_forge.orchestrator.prompt_builder import (
    EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT,
)


def _write_artifacts(
    node_dir: Path, node_id: str, artifacts: List[CodeArtifact]
) -> List[str]:
    """Write artifacts to *node_dir* and return workspace-relative source paths."""
    source_files: List[str] = []
    for artifact in artifacts:
        # The GraphPolyglotMaterializer strips a leading node_id/ prefix when it
        # places files under the node directory. Do the same here for tests.
        rel = Path(artifact.file_path)
        if rel.parts and rel.parts[0] == node_id:
            rel = Path(*rel.parts[1:])
        target = node_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")
        if not artifact.is_header:
            source_files.append(str(rel))
    return source_files


def _test_language_synthesis(
    registry: EmitterRegistry,
    lang: str,
    workspace: Path,
    api_key: str,
) -> Dict[str, Any]:
    """Synthesize an emitter plugin for *lang*, emit source/manifest, and build if possible."""
    print(f"\n=== Testing JIT synthesis for {lang} ===")

    # Remove any previously registered plugin for this language so the test
    # actually exercises the JIT path.
    registry._plugins.pop(lang, None)

    plugin = registry.get_plugin(
        lang,
        synthesize=True,
        boundary_type=BoundaryContract.C_ABI,
    )
    assert isinstance(plugin, PolyglotEmitterPlugin)
    desc = plugin.descriptor
    print(f"descriptor: {desc}")
    assert desc.language_id == lang
    assert BoundaryContract.C_ABI in desc.supported_boundaries
    assert desc.toolchains
    assert desc.file_extensions

    node_id = f"{lang}_kernel"
    node_dir = workspace / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    stubs = registry._artifact_stubs(lang)
    source_artifacts = plugin.emit_source_files(**stubs["source_args"])
    manifest_artifacts = plugin.emit_build_manifest(**stubs["manifest_args"])

    print("emitted source files:")
    for a in source_artifacts:
        print(f"  {a.file_path}: {a.content[:120].replace(chr(10), ' ')}")
    print("emitted manifest files:")
    for a in manifest_artifacts:
        print(f"  {a.file_path}: {a.content[:120].replace(chr(10), ' ')}")

    _write_artifacts(node_dir, node_id, manifest_artifacts)
    source_files = _write_artifacts(node_dir, node_id, source_artifacts)
    node_spec = {
        "lang": lang,
        "toolchain": lang,
        "source_files": source_files,
        "compiler_flags": ["-O3"],
        "exports": ["fast_math_kernel"],
    }

    build_result: Dict[str, Any] = {
        "compiled": False,
        "symbol_callable": False,
        "output": "",
    }
    toolchain_path = shutil.which(lang)
    if lang in ("zig", "go") and toolchain_path:
        print(f"Invoking {lang} toolchain at {toolchain_path}")
        result = SystemToolchainRouter.dispatch_node_build(node_id, node_spec, node_dir)
        build_result["compiled"] = result.returncode == 0
        if lang == "zig":
            so_path = node_dir / f"lib{node_id}.so"
            if so_path.is_file():
                lib = ctypes.CDLL(str(so_path))
                lib.fast_math_kernel.argtypes = [ctypes.c_int64]
                lib.fast_math_kernel.restype = ctypes.c_int64
                val = lib.fast_math_kernel(21)
                print(f"ctypes call fast_math_kernel(21) = {val}")
                build_result["symbol_callable"] = True
                build_result["output"] = str(val)
        elif lang == "go":
            so_path = node_dir / f"{node_id}.so"
            if so_path.is_file():
                lib = ctypes.CDLL(str(so_path))
                lib.fast_math_kernel.argtypes = [ctypes.c_int64]
                lib.fast_math_kernel.restype = ctypes.c_int64
                val = lib.fast_math_kernel(21)
                print(f"ctypes call fast_math_kernel(21) = {val}")
                build_result["symbol_callable"] = True
                build_result["output"] = str(val)
    else:
        print(f"No host toolchain for {lang}; skipping native compilation")

    return build_result


def _test_full_zig_build(api_key: str, workspace: Path, log_path: Path) -> None:
    """Run the full Zig math-kernel prompt through GraphPolyglotMaterializer."""
    print("\n=== Full Zig math-kernel build test ===")
    compiler = IntentCompiler(provider="deepseek", api_key=api_key)
    prompt = (
        "Build a high-performance math kernel in Zig that exports a C-ABI function "
        "fast_math_kernel(x: int64) -> int64. Link this Zig kernel to a Python CLI "
        "via a C-ABI zero-copy bridge. The Python CLI should use ctypes to load the "
        "shared library and call the function."
    )
    graph = compiler.compile_prompt_to_graph(prompt)

    materializer = GraphPolyglotMaterializer(
        workspace,
        llm_provider="deepseek",
        llm_model="deepseek-chat",
        llm_api_key=api_key,
    )
    result = materializer.materialize(graph.model_dump(mode="json"), build=True)
    print(f"materialized: {result['blueprint_path']}")

    build_script = workspace / "build.sh"
    assert build_script.is_file(), "build.sh was not generated"

    proc = subprocess.run(
        ["bash", str(build_script)],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    print(f"build.sh exit code: {proc.returncode}")
    print(f"build.sh stdout: {proc.stdout.strip()}")
    if proc.stderr:
        print(f"build.sh stderr: {proc.stderr.strip()[-500:]}")
    assert proc.returncode == 0, f"build.sh failed: {proc.stderr}"
    assert "fast_math_kernel" in proc.stdout, "Python CLI did not call fast_math_kernel"

    # Verify the shared library was produced.
    zig_node = next(
        (n.node_id for n in graph.nodes if n.lang == "zig"),
        "zig_kernel",
    )
    so_path = workspace / zig_node / f"lib{zig_node}.so"
    assert so_path.is_file(), f"Expected shared library at {so_path}"

    # Verify the accelerator log recorded JIT synthesis and Zig build success.
    log_text = log_path.read_text(encoding="utf-8")
    assert (
        "JIT-synthesized" in log_text
    ), "Accelerator log missing JIT-synthesis success"
    assert (
        "zig build succeeded" in log_text
    ), "Accelerator log missing Zig build success"


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set", file=sys.stderr)
        return 1

    log_path = Path(tempfile.mktemp(suffix=".log"))
    os.environ["AERO_FORGE_ACCEL_LOG"] = str(log_path)
    print(f"Accelerator log: {log_path}")

    # Ensure built-in emitters are loaded into the singleton registry.
    GraphPolyglotMaterializer._ensure_emitters_loaded()
    registry = EmitterRegistry.get_instance()
    registry.configure_jit_synthesis(
        provider="deepseek",
        model="deepseek-chat",
        api_key=api_key,
        prompt=EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT,
    )

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        zig_result = _test_language_synthesis(registry, "zig", workspace, api_key)
        assert zig_result["compiled"], "Zig kernel did not compile"
        assert zig_result["symbol_callable"], "Zig kernel symbol is not callable"

        mojo_result = _test_language_synthesis(registry, "mojo", workspace, api_key)
        # Mojo host compiler is not generally available; source emission is enough.
        mojo_file = workspace / "mojo_kernel" / "src" / "fast_math_kernel.mojo"
        assert mojo_file.is_file(), "Mojo source file was not emitted"

        go_result = _test_language_synthesis(registry, "go", workspace, api_key)
        go_file = workspace / "go_kernel" / "go_kernel" / "src" / "fast_math_kernel.go"
        if not go_file.is_file():
            go_file = workspace / "go_kernel" / "src" / "fast_math_kernel.go"
        assert go_file.is_file(), "Go source file was not emitted"
        if shutil.which("go"):
            assert go_result["compiled"], "Go kernel did not compile"
            assert go_result["symbol_callable"], "Go kernel symbol is not callable"

        _test_full_zig_build(api_key, workspace, log_path)

    print("\nAll universal plugin synthesis checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
