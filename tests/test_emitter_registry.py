"""Tests for the polyglot emitter plugin registry."""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)


class _FakePlugin(PolyglotEmitterPlugin):
    def __init__(self, language_id: str):
        self._language_id = language_id

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id=self._language_id,
            supported_boundaries={BoundaryContract.C_ABI},
            toolchains=[],
            file_extensions=[".fake"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: dict,
        boundary_contracts: list,
    ) -> list:
        return [CodeArtifact(file_path="stub.fake", content="", language="fake")]

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: list,
        compiler_flags: list,
    ) -> CodeArtifact:
        return CodeArtifact(file_path="build.fake", content="", language="fake")


def test_registry_is_singleton() -> None:
    a = EmitterRegistry.get_instance()
    b = EmitterRegistry.get_instance()
    assert a is b


def test_registry_registers_and_looks_up_plugin() -> None:
    registry = EmitterRegistry.get_instance()
    plugin = _FakePlugin("fake_lang")
    registry.register(plugin)
    assert registry.get_plugin("fake_lang") is plugin


def test_registry_finds_emitters_by_boundary() -> None:
    registry = EmitterRegistry.get_instance()
    found = registry.find_emitters_for_boundary(BoundaryContract.C_ABI)
    assert any(p.descriptor.language_id == "fake_lang" for p in found)


def test_registry_builtin_plugins_are_registered() -> None:
    # Importing the emitter modules should register all built-in plugins.
    from aero_forge.builder.emitters import (  # noqa: F401
        cgo_emitter,
        cpp_emitter,
        cs_emitter,
        jni_emitter,
        python_emitter,
        rust_emitter,
    )

    registry = EmitterRegistry.get_instance()
    for lang in ("python", "rust", "cpp", "go", "csharp", "java"):
        plugin = registry.get_plugin(lang)
        assert plugin.descriptor.language_id == lang


def test_registry_concurrent_registration_is_safe() -> None:
    registry = EmitterRegistry.get_instance()
    barrier = threading.Barrier(10)

    def register_and_lookup() -> list:
        barrier.wait()
        registry.register(_FakePlugin(threading.current_thread().name))
        return [registry.get_plugin(threading.current_thread().name).descriptor.language_id]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(register_and_lookup) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 10
