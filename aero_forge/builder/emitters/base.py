"""Base primitives and registry for polyglot source emitter plugins."""

from __future__ import annotations

import builtins
import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, cast

from aero_forge.builder.spec import ASTNode, EngineSpec


class BoundaryContract(Enum):
    """Known cross-language binding contracts."""

    C_ABI = "c_abi"
    PYO3_MATURIN = "pyo3_maturin"
    WASM_WASI = "wasm_wasi"
    JNI = "jni"
    CGO = "cgo"
    PINVOKE = "pinvoke"
    CUDA_HIP_C = "cuda_hip_c"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Describes the targets and toolchains an emitter plugin supports."""

    language_id: str
    supported_boundaries: Set[BoundaryContract]
    toolchains: List[str]
    file_extensions: List[str]
    supports_zero_copy: bool
    supports_async_ffi: bool


@dataclass
class CodeArtifact:
    """A single generated source or build file."""

    file_path: str
    content: str
    language: str
    is_header: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolyglotEmitterPlugin(ABC):
    """Plugin interface for language-specific source emitters."""

    @property
    @abstractmethod
    def descriptor(self) -> CapabilityDescriptor:
        """Return the plugin capability descriptor."""

    @abstractmethod
    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        """Emit source files for *node_spec* under the given *boundary_contracts*."""

    @abstractmethod
    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        """Emit a build-system manifest for the node."""


class EmitterError(Exception):
    """Raised when an emitter cannot render an engine spec."""


class SynthesizedPluginError(EmitterError):
    """Raised when an LLM-synthesized emitter plugin is invalid or unsafe."""


class BaseEmitter(ABC):
    """Render an :class:`EngineSpec` into source code for a target language.

    Subclasses implement language-specific syntax by overriding the
    ``_emit_*`` hooks. The public entry point is :meth:`emit`.
    """

    target_language: str = ""
    indent: str = "    "

    def __init__(self, indent: Optional[str] = None) -> None:
        if indent is not None:
            self.indent = indent
        self._lines: List[str] = []

    def emit(self, spec: EngineSpec) -> str:
        """Return the fully rendered source for *spec*."""
        self._lines = []
        self._emit_preamble(spec)
        self._emit(spec.root, 0)
        self._emit_postamble(spec)
        return "\n".join(self._lines) + "\n"

    # ------------------------------------------------------------------
    # Public hooks for pre/post-amble
    # ------------------------------------------------------------------

    def _emit_preamble(self, spec: EngineSpec) -> None:
        """Hook for file-level headers (imports, pragmas, etc.)."""

    def _emit_postamble(self, spec: EngineSpec) -> None:
        """Hook for file-level footers."""

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _emit(self, node: ASTNode, indent_level: int) -> None:
        handler = getattr(self, f"_emit_{node.kind}", None)
        if handler is None:
            raise EmitterError(
                f"{self.__class__.__name__} has no handler for node kind {node.kind!r}"
            )
        handler(node, indent_level)

    def _emit_children(self, nodes: List[ASTNode], indent_level: int) -> None:
        for child in nodes:
            self._emit(child, indent_level)

    def _emit_block(self, node: ASTNode, indent_level: int) -> None:
        self._emit_children(node.children, indent_level)

    def _write(self, line: str, indent_level: int = 0) -> None:
        self._lines.append(self.indent * indent_level + line)

    def _expr(self, node: ASTNode) -> str:
        """Render an expression node as a single string."""
        return self._emit_expression_to_string(node)

    # ------------------------------------------------------------------
    # Abstract language primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def _emit_module(self, node: ASTNode, indent_level: int) -> None:
        """Render a module / translation unit."""

    @abstractmethod
    def _emit_function(self, node: ASTNode, indent_level: int) -> None:
        """Render a function declaration."""

    @abstractmethod
    def _emit_struct(self, node: ASTNode, indent_level: int) -> None:
        """Render a struct / class / record."""

    @abstractmethod
    def _emit_binding(self, node: ASTNode, indent_level: int) -> None:
        """Render a variable binding / assignment."""

    @abstractmethod
    def _emit_return(self, node: ASTNode, indent_level: int) -> None:
        """Render a return statement."""

    @abstractmethod
    def _emit_import(self, node: ASTNode, indent_level: int) -> None:
        """Render an import / use / include."""

    @abstractmethod
    def _emit_comment(self, node: ASTNode, indent_level: int) -> None:
        """Render a comment line."""

    # ------------------------------------------------------------------
    # Expression helpers (common across emitters)
    # ------------------------------------------------------------------

    def _emit_expression_to_string(self, node: ASTNode) -> str:
        if node.kind == "literal":
            return self._literal(node.value)
        if node.kind == "reference":
            return node.name or "_"
        if node.kind == "call":
            args = ", ".join(self._expr(c) for c in node.children)
            return f"{node.name}({args})"
        if node.kind == "binary_op":
            left, right = node.children
            return f"({self._expr(left)} {node.name} {self._expr(right)})"
        if node.kind == "list":
            return self._list_literal(node.children)
        if node.kind == "dict":
            return self._dict_literal(node.children)
        if node.kind == "param":
            return node.name or "_"
        raise EmitterError(
            f"Unsupported expression kind {node.kind!r} in {self.__class__.__name__}"
        )

    def _literal(self, value: Any) -> str:
        if value is None:
            return self._none_literal()
        if isinstance(value, bool):
            return self._bool_literal(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._string_literal(value)
        if isinstance(value, (list, tuple)):
            from aero_forge.builder.spec import literal

            return self._list_literal([literal(v) for v in value])
        if isinstance(value, dict):
            from aero_forge.builder.spec import literal

            return self._dict_literal(
                [ASTNode(kind="pair", children=[literal(k), literal(v)]) for k, v in value.items()]
            )
        return str(value)

    def _string_literal(self, value: str) -> str:
        return f'"{value}"'

    @abstractmethod
    def _bool_literal(self, value: bool) -> str:
        """Render a boolean literal."""

    @abstractmethod
    def _none_literal(self) -> str:
        """Render a None / null / unit literal."""

    @abstractmethod
    def _list_literal(self, children: List[ASTNode]) -> str:
        """Render a list / vector literal."""

    @abstractmethod
    def _dict_literal(self, pairs: List[ASTNode]) -> str:
        """Render a dict / map literal."""

    @abstractmethod
    def _map_type(self, type_hint: Optional[str]) -> str:
        """Map an abstract type hint to the target language type."""


class EmitterRegistry:
    """Thread-safe singleton registry of :class:`PolyglotEmitterPlugin` instances.

    When the registry is configured with an LLM client and a synthesis prompt, a
    lookup for an unknown language triggers JIT synthesis of a temporary emitter
    plugin. The synthesized class is validated against the base interface and
    the requested FFI boundary contract before it is registered.
    """

    _instance: Optional["EmitterRegistry"] = None
    _registry_lock: Lock = Lock()

    def __new__(cls) -> "EmitterRegistry":
        if cls._instance is None:
            with cls._registry_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "EmitterRegistry":
        return cls()

    def __init__(self) -> None:
        if hasattr(self, "_plugins"):
            return
        self._plugins: Dict[str, PolyglotEmitterPlugin] = {}
        self._lock: Lock = Lock()
        self._synthesis_client: Optional[Any] = None
        self._synthesis_provider: Optional[str] = None
        self._synthesis_model: Optional[str] = None
        self._synthesis_api_key: Optional[str] = None
        self._synthesis_prompt: Optional[str] = None

    def configure_jit_synthesis(
        self,
        llm_client: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Configure the LLM used to synthesize missing emitter plugins."""
        self._synthesis_client = llm_client
        self._synthesis_provider = provider
        self._synthesis_model = model
        self._synthesis_api_key = api_key
        self._synthesis_prompt = prompt

    def register(self, plugin: PolyglotEmitterPlugin) -> None:
        with self._lock:
            key = plugin.descriptor.language_id.lower().strip()
            self._plugins[key] = plugin

    def get_plugin(
        self,
        language_id: str,
        synthesize: bool = True,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> PolyglotEmitterPlugin:
        """Return a registered plugin, synthesizing one on demand if allowed."""
        key = language_id.lower().strip()
        with self._lock:
            if key in self._plugins:
                return self._plugins[key]
            if synthesize and self._can_synthesize():
                plugin = self._synthesize_plugin(key, boundary_type=boundary_type)
                self._plugins[key] = plugin
                return plugin
            raise EmitterError(
                f"No emitter plugin registered for language {language_id!r}. "
                f"Supported: {sorted(self._plugins.keys())}"
            )

    def _can_synthesize(self) -> bool:
        return bool(self._synthesis_prompt) and (
            self._synthesis_client is not None
            or self._synthesis_provider
            or self._synthesis_api_key
        )

    def _get_llm_client(self) -> Any:
        if self._synthesis_client is None:
            from aero_forge.llm.clients import get_llm_client

            self._synthesis_client = get_llm_client(
                provider=self._synthesis_provider or "deepseek",
                model=self._synthesis_model or None,
                api_key=self._synthesis_api_key,
                raise_on_error=True,
            )
            if self._synthesis_client is None:
                raise EmitterError("Could not construct an LLM client for plugin synthesis")
        return self._synthesis_client

    def _synthesize_plugin(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> PolyglotEmitterPlugin:
        """Ask the LLM to generate a temporary emitter plugin for *language_id*."""
        if not self._synthesis_prompt:
            raise EmitterError("JIT synthesis prompt is not configured")

        client = self._get_llm_client()
        user_prompt = self._build_synthesis_user_prompt(language_id, boundary_type)
        messages = [
            {"role": "system", "content": self._synthesis_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = client.generate(messages, temperature=0.2, max_tokens=4096)
        if not raw:
            raise SynthesizedPluginError(f"LLM returned empty plugin source for {language_id!r}")

        code = self._extract_python_code(raw)
        return self._load_and_validate_plugin(code, language_id, boundary_type)

    def _build_synthesis_user_prompt(
        self,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> str:
        boundary = boundary_type.value if boundary_type else "c_abi"
        return (
            f"Synthesize a complete `PolyglotEmitterPlugin` subclass for the language '{language_id}'.\n\n"
            f"The plugin must support the cross-language boundary contract '{boundary}'.\n"
            "Import `PolyglotEmitterPlugin`, `BoundaryContract`, `CapabilityDescriptor`, and "
            "`CodeArtifact` from `aero_forge.builder.emitters.base`. Do NOT redefine these base classes.\n"
            "Create only the concrete emitter class named `<Language>EmitterPlugin` "
            "(e.g. `ZigEmitterPlugin`).\n\n"
            "Implement a `descriptor` property returning a `CapabilityDescriptor` with "
            f"`language_id='{language_id}'`, `supported_boundaries` including `BoundaryContract.{boundary.upper()}`, "
            "and appropriate `toolchains`, `file_extensions`, `supports_zero_copy`, and `supports_async_ffi`.\n"
            "Implement `emit_source_files(node_id, node_spec, boundary_contracts)` and "
            "`emit_build_manifest(node_id, dependencies, compiler_flags)` returning lists of `CodeArtifact`.\n\n"
            "Each entry in `boundary_contracts` is a dict with keys:\n"
            "- `boundary_type` (string, e.g. 'c_abi')\n"
            "- `symbol` (string)\n"
            "- `args` (list of primitive type names: 'int32', 'int64', 'float32', 'float64', 'pointer')\n"
            "- `return_type` (primitive type name or empty string for void)\n"
            "- `is_zero_copy` (boolean)\n\n"
            "The source files must contain a real, exported function for the first contract.\n"
            "For C-ABI use `export fn` (Zig), `#[no_mangle] pub extern \"C\" fn` (Rust), "
            "`extern \"C\"` (C/C++), `//export` (Go), `[UnmanagedCallersOnly]` (C#), or JNI signatures (Java).\n"
            "For Zig, mark every parameter as used (e.g. `_ = arg_0;`) and return `return;` for void "
            "or a typed literal such as `return @as(i64, 42);` for non-void returns.\n\n"
            "For the build manifest, return a minimal valid file (build.zig, Makefile, etc.). "
            "Use `.format()` or simple string concatenation for complex multi-line strings; do NOT put "
            "backslash escapes inside f-string expression braces because that is a Python syntax error.\n\n"
            "Return ONLY valid Python code inside a single markdown ```python ... ``` block. No prose."
        )

    @staticmethod
    def _extract_python_code(raw: str) -> str:
        """Extract Python source from a fenced code block, or return the raw text."""
        fenced = re.findall(r"```(?:python)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if fenced:
            return fenced[-1].strip()
        return raw.strip()

    @staticmethod
    def _strip_redefined_helpers(code: str) -> str:
        """Remove any local redefinitions of the shared base classes.

        LLM output often includes self-contained dataclasses/enums. We already
        inject the real classes into the execution namespace, so the generated
        emitter only needs to subclass and instantiate them.
        """
        helper_names = ["BoundaryContract", "CapabilityDescriptor", "CodeArtifact", "PolyglotEmitterPlugin"]
        pattern = re.compile(
            r"^(class\s+(?:" + "|".join(helper_names) + r")\b[^:\n]*:[^\n]*\n(?:\s+.*\n|\n)*)",
            re.MULTILINE,
        )
        code = pattern.sub("", code)
        # Remove dangling @abstractmethod / @property decorators that may be left behind.
        code = re.sub(r"^\s*@(abstractmethod|property|dataclass|staticmethod)\s*\n", "", code, flags=re.MULTILINE)
        return code

    def _load_and_validate_plugin(
        self,
        code: str,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> PolyglotEmitterPlugin:
        """Exec the generated plugin source, validate it, and return an instance."""
        namespace: Dict[str, Any] = {
            "__builtins__": builtins,
            "PolyglotEmitterPlugin": PolyglotEmitterPlugin,
            "BoundaryContract": BoundaryContract,
            "CapabilityDescriptor": CapabilityDescriptor,
            "CodeArtifact": CodeArtifact,
            "List": List,
            "Dict": Dict,
            "Set": Set,
            "Optional": Optional,
            "Any": Any,
        }
        # Allow the generated code to import from the aero_forge package if it wishes.
        try:
            exec(code, namespace)  # nosec B102
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Generated plugin source for {language_id!r} could not be executed: {exc}"
            ) from exc

        # The generated source may re-declare helper base classes. We identify
        # the concrete emitter by name (<Language>EmitterPlugin) and then verify
        # its descriptor and emitted artifacts against the real base class.
        candidates = [
            obj
            for name, obj in namespace.items()
            if isinstance(obj, type)
            and name.lower().endswith("emitterplugin")
            and name.lower() != "polyglotemitterplugin"
        ]
        if not candidates:
            raise SynthesizedPluginError(
                f"Generated plugin source for {language_id!r} did not define a "
                "concrete `*EmitterPlugin` subclass"
            )

        cls = candidates[0]
        try:
            raw_instance = cls()
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Could not instantiate generated plugin {cls.__name__!r}: {exc}"
            ) from exc

        instance = self._wrap_plugin(raw_instance)
        self._validate_plugin(instance, language_id, boundary_type)
        return instance

    def _wrap_plugin(self, raw_instance: Any) -> PolyglotEmitterPlugin:
        """Return a plugin instance whose descriptor and artifacts are canonical types.

        LLM output may redefine ``CapabilityDescriptor``/``CodeArtifact`` inside
        the execution namespace. We create a small wrapper subclass so the
        ``descriptor`` property and emission methods always return the real
        base classes.
        """

        def _norm_boundary(value: Any) -> BoundaryContract:
            if isinstance(value, BoundaryContract):
                return value
            raw = value.value if hasattr(value, "value") else str(value)
            return BoundaryContract(raw)

        def _norm_descriptor(value: Any) -> CapabilityDescriptor:
            if isinstance(value, CapabilityDescriptor):
                return value
            supported: Set[BoundaryContract] = set()
            for item in value.supported_boundaries:
                supported.add(_norm_boundary(item))
            return CapabilityDescriptor(
                language_id=str(value.language_id),
                supported_boundaries=supported,
                toolchains=list(value.toolchains),
                file_extensions=list(value.file_extensions),
                supports_zero_copy=bool(value.supports_zero_copy),
                supports_async_ffi=bool(value.supports_async_ffi),
            )

        def _norm_artifact(value: Any) -> CodeArtifact:
            if isinstance(value, CodeArtifact):
                return value
            return CodeArtifact(
                file_path=str(value.file_path),
                content=str(value.content),
                language=str(value.language),
                is_header=bool(getattr(value, "is_header", False)),
                metadata=dict(getattr(value, "metadata", {})),
            )

        try:
            raw_descriptor = raw_instance.descriptor
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Generated plugin {raw_instance.__class__.__name__!r} raised from `descriptor`: {exc}"
            ) from exc

        norm_descriptor = _norm_descriptor(raw_descriptor)
        orig_source = raw_instance.emit_source_files
        orig_manifest = raw_instance.emit_build_manifest

        def emit_source_files(self, *args: Any, **kwargs: Any) -> List[CodeArtifact]:
            result = orig_source(*args, **kwargs)
            if not isinstance(result, list):
                result = [result]
            return [_norm_artifact(a) for a in result]

        def emit_build_manifest(self, *args: Any, **kwargs: Any) -> List[CodeArtifact]:
            result = orig_manifest(*args, **kwargs)
            if isinstance(result, CodeArtifact):
                result = [result]
            return [_norm_artifact(a) for a in result]

        wrapper_cls = type(
            raw_instance.__class__.__name__,
            (raw_instance.__class__,),
            {
                "descriptor": property(lambda self: norm_descriptor),
                "emit_source_files": emit_source_files,
                "emit_build_manifest": emit_build_manifest,
            },
        )
        return cast(PolyglotEmitterPlugin, wrapper_cls())

    def _validate_plugin(
        self,
        plugin: PolyglotEmitterPlugin,
        language_id: str,
        boundary_type: Optional[BoundaryContract] = None,
    ) -> None:
        """Verify that a synthesized plugin satisfies the base contract and boundary."""
        try:
            descriptor = plugin.descriptor
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} raised an error from `descriptor`: {exc}"
            ) from exc

        if not isinstance(descriptor, CapabilityDescriptor):
            raise SynthesizedPluginError(
                f"Synthesized plugin descriptor for {language_id!r} is not a CapabilityDescriptor"
            )

        if descriptor.language_id.lower() != language_id.lower():
            raise SynthesizedPluginError(
                f"Synthesized plugin claims language_id={descriptor.language_id!r} "
                f"but {language_id!r} was requested"
            )

        if not isinstance(descriptor.supported_boundaries, (set, frozenset, list, tuple)):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `supported_boundaries` is not a collection"
            )
        supported = set(descriptor.supported_boundaries)
        for item in supported:
            if not isinstance(item, BoundaryContract):
                raise SynthesizedPluginError(
                    f"Synthesized plugin for {language_id!r} has invalid boundary {item!r}"
                )

        if boundary_type is not None and boundary_type not in supported:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not support {boundary_type.value!r}"
            )

        if not descriptor.toolchains or not descriptor.file_extensions:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} must provide non-empty "
                "`toolchains` and `file_extensions`"
            )

        if not callable(getattr(plugin, "emit_source_files", None)):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not implement `emit_source_files`"
            )
        if not callable(getattr(plugin, "emit_build_manifest", None)):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} does not implement `emit_build_manifest`"
            )

        # Type-check the emitted artifacts by calling the methods with minimal stubs.
        try:
            stubs = self._artifact_stubs(language_id)
            source_artifacts = plugin.emit_source_files(**stubs["source_args"])
            manifest = plugin.emit_build_manifest(**stubs["manifest_args"])
        except Exception as exc:
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} failed during artifact emission: {exc}"
            ) from exc

        if not isinstance(source_artifacts, list) or not all(
            isinstance(a, CodeArtifact) for a in source_artifacts
        ):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `emit_source_files` must return a list of CodeArtifact"
            )
        if not isinstance(manifest, list) or not all(
            isinstance(a, CodeArtifact) for a in manifest
        ):
            raise SynthesizedPluginError(
                f"Synthesized plugin for {language_id!r} `emit_build_manifest` must return a list of CodeArtifact"
            )

    @staticmethod
    def _artifact_stubs(language_id: str) -> Dict[str, Any]:
        return {
            "source_args": {
                "node_id": f"{language_id}_kernel",
                "node_spec": {
                    "lang": language_id,
                    "toolchain": language_id,
                    "source_files": [f"src/{language_id}_kernel.zig"],
                    "compiler_flags": ["-O3"],
                    "exports": ["fast_math_kernel"],
                },
                "boundary_contracts": [
                    {
                        "boundary_type": "c_abi",
                        "boundary": "c_abi",
                        "symbol": "fast_math_kernel",
                        "args": ["int64"],
                        "return_type": "int64",
                        "is_zero_copy": True,
                    }
                ],
            },
            "manifest_args": {
                "node_id": f"{language_id}_kernel",
                "dependencies": [],
                "compiler_flags": ["-O3"],
            },
        }

    def find_emitters_for_boundary(self, boundary: BoundaryContract) -> List[PolyglotEmitterPlugin]:
        with self._lock:
            return [
                plugin
                for plugin in self._plugins.values()
                if boundary in plugin.descriptor.supported_boundaries
            ]


# Re-export `literal` for emitter internal use.
from aero_forge.builder.spec import literal  # noqa: E402,F401
