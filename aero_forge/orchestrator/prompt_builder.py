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
        self.system_message = (
            system_message
            or (
                "You are an expert Python and Rust programmer. Fix the provided function so it compiles and passes its tests. "
                "If the code already compiles but tests fail, correct the algorithm based on the test output. "
                "If the failure is an IndexError or any out-of-order execution issue, ensure all lists, tuples, dictionaries, and data structures are fully initialized and populated before any calculation or indexing operation. "
                "Do not invoke methods or access indices before the structure has been built. "
                "Return ONLY the corrected function definition (no markdown fences, no explanation)."
            )
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

    The prompt explicitly instructs the model to emit a polyglot
    ``hybrid_rust_python`` blueprint whenever the user intent involves both
    Python and Rust/PyO3/Maturin/FFI, and includes a concrete few-shot example.
    """
    import re

    explicit_files = re.findall(r"\b[A-Za-z_][\w/]*\.py\b", prompt)
    explicit_files_section = (
        "\n".join(["Explicitly requested files (include these exact paths in the manifest):"]
                   + [f"  - {f}" for f in explicit_files])
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
    parts.extend(["", "Return ONLY the YAML blueprint.aero. No markdown fences, no explanation."])
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
Your task is to generate a complete, self-contained Python class that subclasses `PolyglotEmitterPlugin` for a programming language requested by the user.

Base class contract (you may copy this structure):

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set

class BoundaryContract(Enum):
    C_ABI = "c_abi"
    PYO3_MATURIN = "pyo3_maturin"
    WASM_WASI = "wasm_wasi"
    JNI = "jni"
    CGO = "cgo"
    PINVOKE = "pinvoke"
    CUDA_HIP_C = "cuda_hip_c"

@dataclass(frozen=True)
class CapabilityDescriptor:
    language_id: str
    supported_boundaries: Set[BoundaryContract]
    toolchains: List[str]
    file_extensions: List[str]
    supports_zero_copy: bool
    supports_async_ffi: bool

@dataclass
class CodeArtifact:
    file_path: str
    content: str
    language: str
    is_header: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PolyglotEmitterPlugin(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> CapabilityDescriptor: ...
    @abstractmethod
    def emit_source_files(self, node_id: str, node_spec: Dict[str, Any], boundary_contracts: List[Dict[str, Any]]) -> List[CodeArtifact]: ...
    @abstractmethod
    def emit_build_manifest(self, node_id: str, dependencies: List[str], compiler_flags: List[str]) -> List[CodeArtifact]: ...
```

Rules:
1. Return ONLY valid Python code inside a single markdown ```python ... ``` block.
2. The class name must be `<Language>EmitterPlugin` (e.g. `ZigEmitterPlugin`).
3. The `descriptor` property must return a `CapabilityDescriptor` whose `language_id` matches the requested language.
4. `supported_boundaries` must be a set of `BoundaryContract` values and must include the boundary contract requested in the user prompt.
5. `toolchains` and `file_extensions` must be non-empty lists.
6. `emit_source_files` must return a list of `CodeArtifact` objects. The source files must implement a real exported function matching the first entry in `boundary_contracts` (use its `symbol`, `args`, and `return_type`).
7. For C-ABI boundaries, export functions using the correct visibility for the target language:
   - Zig: `export fn symbol(...) ...`
   - Rust: `#[no_mangle] pub extern "C" fn symbol(...)`
   - C/C++: `extern "C"` block
   - Go: `//export symbol` above `func symbol(...)`
   - C#: `[UnmanagedCallersOnly]`
   - Java: JNI `JNIEXPORT ... JNICALL` signature
8. `emit_build_manifest` must return a list of `CodeArtifact` objects for build files such as `Cargo.toml`, `CMakeLists.txt`, `build.zig`, `package.json`, `setup.py`, or a `Makefile`.
9. Do not write placeholder comments, TODOs, or unimplemented stubs. Every emitted artifact must contain valid, compilable source or build configuration.
"""


__all__ = [
    "PromptBuilder",
    "build_blueprint_plan_prompt",
    "EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT",
]
