"""Polyglot generation core for aero-forge."""

from __future__ import annotations

from aero_forge.builder.artifact_generator import ArtifactGenerator
from aero_forge.builder.builder import BuildOutput, build_engine
from aero_forge.builder.emitters import get_emitter
from aero_forge.builder.language_router import resolve_target_language
from aero_forge.builder.spec import (
    ASTNode,
    EngineSpec,
    binding,
    binary_op,
    block,
    call,
    comment,
    dict_literal,
    field,
    function,
    get_type,
    import_node,
    list_literal,
    literal,
    module,
    param,
    reference,
    return_node,
    set_type,
    spec_from_python,
    struct,
)

__all__ = [
    "ArtifactGenerator",
    "BuildOutput",
    "EngineSpec",
    "ASTNode",
    "build_engine",
    "get_emitter",
    "resolve_target_language",
    "module",
    "function",
    "param",
    "binding",
    "return_node",
    "call",
    "binary_op",
    "literal",
    "reference",
    "struct",
    "field",
    "import_node",
    "comment",
    "block",
    "list_literal",
    "dict_literal",
    "get_type",
    "set_type",
    "spec_from_python",
]
