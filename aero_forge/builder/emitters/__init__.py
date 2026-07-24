"""Polyglot target emitters for aero-forge engine specs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from aero_forge.builder.emitters.base import BaseEmitter, EmitterError
from aero_forge.builder.emitters.cpp_emitter import CppEmitter
from aero_forge.builder.emitters.python_emitter import PythonEmitter
from aero_forge.builder.emitters.rust_emitter import RustEmitter


class EmitterRegistry:
    """Central registry mapping language names to emitter classes."""

    _registry: Dict[str, Type[BaseEmitter]] = {}

    @classmethod
    def register(cls, language: str, emitter_cls: Type[BaseEmitter]) -> None:
        cls._registry[language.lower()] = emitter_cls

    @classmethod
    def get(cls, language: str) -> Optional[Type[BaseEmitter]]:
        return cls._registry.get(language.lower())

    @classmethod
    def supported_languages(cls) -> List[str]:
        return sorted(cls._registry.keys())


EmitterRegistry.register("rust", RustEmitter)
EmitterRegistry.register("python", PythonEmitter)
EmitterRegistry.register("cpp", CppEmitter)


def get_emitter(language: str) -> BaseEmitter:
    """Return an emitter instance for *language*.

    Raises :class:`EmitterError` when no emitter is registered.
    """
    emitter_cls = EmitterRegistry.get(language)
    if emitter_cls is None:
        raise EmitterError(
            f"No emitter registered for target language {language!r}. "
            f"Supported: {EmitterRegistry.supported_languages()}"
        )
    return emitter_cls()


__all__ = [
    "BaseEmitter",
    "CppEmitter",
    "EmitterError",
    "EmitterRegistry",
    "PythonEmitter",
    "RustEmitter",
    "get_emitter",
]
