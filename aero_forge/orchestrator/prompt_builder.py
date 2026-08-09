"""Build consolidated LLM prompts from accumulated build/test errors."""

from __future__ import annotations

import re
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
Your ONLY task is to COMPLETE the provided skeleton and return a working Python class named `__LANGUAGE_TITLE__EmitterPlugin` that subclasses `PolyglotEmitterPlugin`.

You will be asked to emit a function with this exact signature:

__FUNCTION_SIGNATURE__

Semantic purpose of the emitted function:
__FUNCTION_CONTEXT__

You MUST:
1. Return ONLY the completed class between the delimiters `__AERO_LOGIC_START__` and `__AERO_LOGIC_END__`. Do not output prose, markdown headers, explanations, conversational text, or code commentary outside those delimiters.
2. Fill in EVERY section marked with `__AERO_IN_FILL__` or `# AERO-TODO`. Do not leave any placeholder logic in `emit_source_files`.
3. Implement `descriptor`, `emit_source_files`, and `emit_build_manifest`.
4. The `emit_source_files` method MUST emit real, compilable source code for `__LANGUAGE_ID__` that matches the exact signature above and implements the semantic purpose. Do NOT leave placeholder expressions like `arg_0 * 2` unless the semantic purpose explicitly says so.
5. If the semantic purpose is empty or vague, implement a sensible numeric operation that matches the signature and returns a meaningful scalar.
6. Use the appropriate language syntax:
   - Zig: `export fn` with `i64`, `f64`, `[*c]f64`, etc. An `export fn` returning a scalar C type CANNOT use `try`; handle fallible calls with `catch (return 0)` or `catch unreachable`.
   - Go: `//export` + `import "C"` and use `C.longlong`, `C.double`, `*C.double`.
   - C/C++: `extern "C"` with `int64_t`, `double`, `double*`.
   - C#: `[UnmanagedCallersOnly(EntryPoint = "...")]`.
   - Java: JNI signatures.
   - Mojo: `fn` with `Int64`, `Float64`, `DTypePointer[DType.float64]`.
7. `toolchains` and `file_extensions` must be non-empty lists.
8. Do not redeclare `BoundaryContract`, `CapabilityDescriptor`, `CodeArtifact`, or `PolyglotEmitterPlugin`.
9. Do not write TODO comments, placeholder comments, or unimplemented stubs outside the fill markers.
10. The response must start exactly with `__AERO_LOGIC_START__` on its own line and end exactly with `__AERO_LOGIC_END__` on its own line.

Skeleton to complete:

__AERO_LOGIC_START__
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
            file_extensions=["__EXT__"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(self, node_id, node_spec, boundary_contracts):
        contract = boundary_contracts[0]
        symbol = contract["symbol"]
        args = contract["args"]
        return_type = contract.get("return_type", "int64") or "int64"

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
        elif return_type == "int64":
            ret = "i64"
        elif return_type == "float32":
            ret = "f32"
        elif return_type == "float64":
            ret = "f64"
        else:
            ret = "void"

        # The body below must implement the exact function signature:
        #   __FUNCTION_SIGNATURE__
        # Use ONLY the parameters in arg_names ({arg_list}).
        # Do not introduce arg_1, arg_2, or other undeclared identifiers.
        __AERO_IN_FILL__

    def emit_build_manifest(self, node_id, dependencies, compiler_flags):
        manifest = __MANIFEST_CONTENT__
        return [
            CodeArtifact(
                file_path="__MANIFEST_FILE__",
                content=manifest,
                language="__LANGUAGE_ID__",
            )
        ]
__AERO_LOGIC_END__

Required substitutions (already filled in the prompt you receive):
- `__LANGUAGE_TITLE__`, `__LANGUAGE_ID__`, `__BOUNDARY_NAME__`, `__TOOLCHAIN__`, `__EXT__`, `__MANIFEST_FILE__`, `__MANIFEST_CONTENT__`, `__FUNCTION_SIGNATURE__`, `__FUNCTION_CONTEXT__`.
"""


def extract_aero_logic(raw: str) -> str:
    """Return the payload delimited by ``__AERO_LOGIC_START__`` / ``__AERO_LOGIC_END__``.

    The Aero-Forge Structured Synthesis Payload requires the model to wrap every
    machine-readable response in these markers. If the markers are present, all
    surrounding prose and markdown are ignored. Falls back to markdown fenced
    code blocks, then to the trimmed raw text.
    """
    if not raw:
        return ""
    text = raw.strip()

    start_match = re.search(
        r"__AERO_LOGIC_START__\s*\r?\n?",
        text,
        re.IGNORECASE,
    )
    end_match = re.search(
        r"\r?\n?\s*__AERO_LOGIC_END__",
        text,
        re.IGNORECASE,
    )
    if start_match:
        if end_match and end_match.start() >= start_match.end():
            return text[start_match.end():end_match.start()].strip()
        # Truncated response: keep everything after the start marker so the
        # fence parser can still extract partial code.
        return text[start_match.end():].strip()

    # Markdown fence fallback (first fenced block is the primary implementation).
    fenced = re.findall(
        r"```\s*(?:\w*)\s*(?::\s*[^\n\r]*?)?\s*\r?\n([\s\S]*?)\r?\n?\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced[0].strip()

    return text


__all__ = [
    "PromptBuilder",
    "build_blueprint_plan_prompt",
    "EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT",
    "extract_aero_logic",
]
