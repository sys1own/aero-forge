"""Build consolidated LLM prompts from accumulated build/test errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aero_forge.blueprint_templates import load_all_templates
from aero_forge.prompts import (
    BLUEPRINT_PLAN_INSTRUCTIONS,
    POLYGLOT_BLUEPRINT_EXAMPLE,
)


class PromptBuilder:
    """Accumulate error context and produce a single repair prompt."""

    def __init__(self, system_message: Optional[str] = None):
        self.system_message = system_message or (
            "You are an expert Python and Rust programmer. Fix the provided function so it compiles and passes its tests. "
            "If the code already compiles but tests fail, correct the algorithm based on the test output. "
            "If the failure is an IndexError or any out-of-order execution issue, ensure all lists, tuples, dictionaries, and data structures are fully initialized and populated before any calculation or indexing operation. "
            "Do not invoke methods or access indices before the structure has been built. "
            "Return ONLY the corrected function definition (no markdown fences, no explanation)."
        )
        self.errors: List[str] = []

    def add_error(self, error: str) -> None:
        """Add an error message to the context."""
        if error and error not in self.errors:
            self.errors.append(error)

    def clear(self) -> None:
        self.errors.clear()

    def build(
        self,
        function_name: str,
        function_source: str,
        additional_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of chat-completion messages."""
        parts = [
            f"Fix the Python function `{function_name}` so that it compiles and passes its tests.",
            "",
            f"Function `{function_name}`:",
            function_source,
        ]
        if self.errors:
            parts.extend(["", "Accumulated failures:"])
            for idx, err in enumerate(self.errors, 1):
                parts.append(f"[{idx}] {err}")
        if additional_context:
            parts.extend(["", "Additional context:", additional_context])
        parts.extend(["", "Return ONLY the corrected function definition."])
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": "\n".join(parts)},
        ]


def build_blueprint_plan_prompt(
    prompt: str,
    project_name: str,
    constraints: Optional[str] = None,
    intent: Optional[str] = None,
    correction_context: Optional[str] = None,
) -> str:
    """Return the planning prompt for generating a ``blueprint.aero``.

    The prompt instructs the model to emit a universal ``graph_polyglot``
    blueprint. The model may propose any language/toolchain (Go, C#, Java, Zig,
    Mojo, etc.) because missing ``PolyglotEmitterPlugin`` modules are synthesized
    and validated on demand.
    """
    import re

    explicit_files = re.findall(r"\b[A-Za-z_][\w/]*\.py\b", prompt)
    explicit_files_section = (
        "\n".join(
            ["Explicitly requested files (include these exact paths in the manifest):"]
            + [f"  - {f}" for f in explicit_files]
        )
        if explicit_files
        else ""
    )

    parts = [
        BLUEPRINT_PLAN_INSTRUCTIONS,
        "",
        "Reference blueprint.aero templates (use them as structural guides):",
        load_all_templates(),
        "",
        POLYGLOT_BLUEPRINT_EXAMPLE,
        "",
        f"User intent classified as: {intent or 'unspecified'}. "
        "Respect that intent when choosing architecture and toolchains.",
        "",
        f"Project: {project_name}",
        f"Prompt: {prompt}",
        f"Constraints: {constraints or 'none'}",
    ]
    if explicit_files_section:
        parts.extend(["", explicit_files_section])
    parts.extend(
        ["", "Return ONLY the YAML blueprint.aero. No markdown fences, no explanation."]
    )
    if correction_context:
        parts.extend(
            [
                "",
                "CORRECTION REQUIRED:",
                correction_context,
            ]
        )
    return "\n".join(parts)


EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT = """You are the Aero-Forge Emitter Synthesis Agent.
Your ONLY task is to generate a complete, working Python class that subclasses `PolyglotEmitterPlugin`.

You MUST use the following skeleton, replace every placeholder, and return the completed class inside a single markdown ```python ... ``` block. Do not write prose.

Skeleton:

```python
from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    PolyglotEmitterPlugin,
)


class __LANGUAGE_TITLE__EmitterPlugin(PolyglotEmitterPlugin):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="__LANGUAGE_ID__",
            supported_boundaries={BoundaryContract.__BOUNDARY_NAME__},
            toolchains=["__TOOLCHAIN__"],
            file_extensions=[".__EXT__"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(self, node_id, node_spec, boundary_contracts):
        contract = boundary_contracts[0]
        symbol = contract["symbol"]
        args = contract["args"]
        return_type = contract["return_type"]

        arg_names = [f"arg_{i}" for i in range(len(args))]
        arg_decls = []
        for name, kind in zip(arg_names, args):
            if kind == "int32":
                arg_decls.append(f"{name}: i32")
            elif kind == "int64":
                arg_decls.append(f"{name}: i64")
            elif kind == "float32":
                arg_decls.append(f"{name}: f32")
            elif kind == "float64":
                arg_decls.append(f"{name}: f64")
            elif kind == "pointer":
                arg_decls.append(f"{name}: [*c]f64")
            else:
                arg_decls.append(f"{name}: i64")
        arg_list = ", ".join(arg_decls)

        if return_type == "int32":
            ret = "i32"
            ret_expr = "arg_0 * 2"
        elif return_type == "int64":
            ret = "i64"
            ret_expr = "arg_0 * 2"
        elif return_type == "float32":
            ret = "f32"
            ret_expr = "arg_0 * 2.0"
        elif return_type == "float64":
            ret = "f64"
            ret_expr = "arg_0 * 2.0"
        else:
            ret = "void"
            ret_expr = ""

        # This skeleton uses Zig syntax as an example. Replace the function body
        # with the correct syntax for the requested language (`__LANGUAGE_ID__`).
        body_lines = [f"export fn {symbol}({arg_list}) {ret} {{"]
        for name in arg_names:
            if name not in ret_expr:
                body_lines.append(f"    _ = {name};")
        if ret_expr:
            body_lines.append(f"    return @as({ret}, {ret_expr});")
        else:
            body_lines.append("    return;")
        body_lines.append("}")
        source = "\\n".join(body_lines)

        return [
            CodeArtifact(
                file_path=f"{node_id}/src/{symbol}.__EXT__",
                content=source,
                language="__LANGUAGE_ID__",
            )
        ]

    def emit_build_manifest(self, node_id, dependencies, compiler_flags):
        manifest = __MANIFEST_CONTENT__
        return [
            CodeArtifact(
                file_path="__MANIFEST_FILE__",
                content=manifest,
                language="__LANGUAGE_ID__",
            )
        ]
```

Required substitutions:
1. Replace `__LANGUAGE_TITLE__` with the PascalCase language name (e.g. Zig, Go, Mojo).
2. Replace `__LANGUAGE_ID__` with the requested lowercase language id (e.g. zig, go, mojo).
3. Replace `__BOUNDARY_NAME__` with the boundary contract enum name requested by the user (e.g. C_ABI, CGO).
4. Replace `__TOOLCHAIN__` with the canonical build command (e.g. `zig`, `go`, `mojo`, `gcc`).
5. Replace `__EXT__` with the language's source file extension (e.g. `zig`, `go`, `mojo`, `c`, `cs`, `java`).
6. Replace `__MANIFEST_FILE__` with the build file name (e.g. `build.zig`, `go.mod`, `Makefile`, `build.sh`, `pom.xml`, `CMakeLists.txt`).
7. Replace `__MANIFEST_CONTENT__` with a minimal, valid build-system file body for that language.

Rules:
- The function body in `emit_source_files` MUST be real code for `__LANGUAGE_ID__`, not the Zig example left unchanged. Use `export fn` for Zig, `//export`+`import "C"` for Go, `extern "C"` for C/C++, `[UnmanagedCallersOnly]` for C#, and JNI signatures for Java.
- The emitted function must match the first boundary contract's `symbol`, `args` types (`int32`, `int64`, `float32`, `float64`, `pointer`), and `return_type`.
- `toolchains` and `file_extensions` must be non-empty lists.
- Do not write placeholder comments, TODOs, or unimplemented stubs. Every artifact must contain valid, compilable source or build configuration.
- Return ONLY valid Python code inside a single markdown ```python ... ``` block. No prose.
"""


__all__ = [
    "PromptBuilder",
    "build_blueprint_plan_prompt",
    "EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT",
]
