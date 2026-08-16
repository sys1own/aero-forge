"""Rust target emitter for aero-forge engine specs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.builder.artifact_generator import Artifact, ArtifactBundle
from aero_forge.builder.emitters.base import (
    BaseEmitter,
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)
from aero_forge.builder.spec import ASTNode, EngineSpec, function, module, param


class SignatureBodyMatch:
    """Validator that reconciles a function's declared Rust return type with
    the type realized by its body."""

    @staticmethod
    def _is_void(t: Optional[str]) -> bool:
        return (t or "").strip() in ("", "void", "None", "()")

    @staticmethod
    def _is_numeric(t: Optional[str]) -> bool:
        return (t or "").strip() in {
            "i8", "i16", "i32", "i64", "i128",
            "u8", "u16", "u32", "u64", "u128",
            "isize", "usize", "f32", "f64",
        }

    @staticmethod
    def _is_pyoish(t: Optional[str]) -> bool:
        t = (t or "").strip()
        return t.startswith("PyResult<") or "Py" in t or "Python" in t

    @staticmethod
    def _is_complex(t: Optional[str]) -> bool:
        """Return True for generic/container types we cannot reliably infer from a single expression."""
        t = (t or "").strip()
        return any(
            s in t
            for s in (
                "Vec<", "HashMap<", "HashSet<", "BTreeMap<", "BTreeSet<",
                "Option<", "Result<", "Box<", "Arc<", "Rc<", "RefCell<",
                "String",
            )
        )

    @classmethod
    def unify(cls, declared: Optional[str], inferred: Optional[str]) -> Optional[str]:
        """Return the return type that the signature should use.

        * A non-void, PyO3/FFI return type is preserved exactly.
        * If the two primitive numeric types differ, the declared type wins and
          the final return expression is cast at emission time.
        * If the declared type is void, the body's inferred type is used.
        * Otherwise the body type wins so the signature matches the actual
          returned value.
        """
        if cls._is_void(declared):
            return inferred
        if cls._is_void(inferred):
            return declared
        declared = declared.strip()
        inferred = inferred.strip()
        if declared == inferred:
            return declared
        if cls._is_pyoish(declared) or cls._is_complex(declared):
            return declared
        if cls._is_numeric(declared) and cls._is_numeric(inferred):
            return declared
        # Body type wins so the emitted code compiles without guessing a cast.
        return inferred

    @classmethod
    def validate(cls, declared: Optional[str], inferred: Optional[str]) -> None:
        """Raise a descriptive error when types are completely incompatible."""
        if cls._is_void(declared) or cls._is_void(inferred):
            return
        if declared.strip() == inferred.strip():
            return
        if (
            cls._is_pyoish(declared)
            or cls._is_complex(declared)
            or (cls._is_numeric(declared) and cls._is_numeric(inferred))
        ):
            return
        raise ValueError(
            f"Return-type mismatch: declared {declared!r} but body returns {inferred!r}"
        )


class RustEmitter(BaseEmitter):
    """Emit syntactically valid Rust source from an engine spec."""

    target_language = "rust"

    # Tracks extra Rust module files requested by the spec metadata.
    module_sources: Dict[str, str]
    module_artifacts: ArtifactBundle

    def __init__(self, indent: Optional[str] = None) -> None:
        super().__init__(indent=indent)
        self.module_sources = {}
        self.module_artifacts = ArtifactBundle()
        self._current_return_type: Optional[str] = None

    def emit(self, spec: EngineSpec) -> str:
        """Return the fully rendered ``src/lib.rs`` source for *spec*.

        If ``spec.metadata["module_files"]`` is provided, each submodule file
        is rendered into :attr:`module_sources` and registered inside ``lib.rs``
        via ``mod`` declarations and a ``#[pymodule]`` registration block.
        """
        self._lines = []
        self.module_sources = {}
        self.module_artifacts = ArtifactBundle(
            metadata={"project_name": spec.name, "language": "rust"}
        )

        self.is_pyo3 = self._is_pyo3_spec(spec)
        self.is_c_abi = self._is_c_abi_spec(spec)
        self.uses_numpy = self._uses_numpy(spec)
        self.uses_rayon = self._uses_rayon(spec)

        module_files = spec.metadata.get("module_files") or []
        root_functions = [c for c in spec.root.children if c.kind == "function"]
        module_registrations: List[Tuple[str, str]] = []

        for mf in module_files:
            mod_root = mf.get("root") or ASTNode(
                kind="module", name="module", children=[]
            )
            mod_path = str(mf.get("path", "src/mod.rs"))
            mod_name = self._module_name_from_path(mod_path)
            mod_source = self._emit_module_file(mod_root, mod_name)
            self.module_sources[mod_path] = mod_source
            self.module_artifacts.artifacts.append(
                Artifact(path=mod_path, content=mod_source)
            )
            for child in mod_root.children:
                if child.kind == "function" and child.name:
                    module_registrations.append((mod_name, child.name))

        self._emit_preamble(spec)

        # Submodule declarations come before the lib's own items.
        for mod_path in self.module_sources:
            mod_name = self._module_name_from_path(mod_path)
            self._write(f"pub mod {mod_name};")
        if self.module_sources:
            self._write("")

        self._emit(spec.root, 0)

        if self.is_pyo3 and (root_functions or module_registrations):
            self._emit_pymodule_block(spec, root_functions, module_registrations)

        self._emit_postamble(spec)
        return "\n".join(self._lines) + "\n"

    def emit_artifacts(self) -> ArtifactBundle:
        """Return any submodule artifacts produced during the last ``emit`` call."""
        return self.module_artifacts

    # ------------------------------------------------------------------
    # Preamble / postamble
    # ------------------------------------------------------------------

    def _emit_preamble(self, spec: EngineSpec) -> None:
        self._write("// Auto-generated by aero-forge polyglot emitter.")
        if self.is_pyo3:
            self._write("use pyo3::prelude::*;")
            if self.uses_numpy:
                self._write("use numpy::PyArray2;")
        if self.uses_rayon:
            self._write("use rayon::prelude::*;")
        if self.uses_numpy and not self.is_pyo3:
            self._write("use ndarray::{Array2, ArrayView2};")
        self._write("")

    # ------------------------------------------------------------------
    # Language-specific emission hooks
    # ------------------------------------------------------------------

    def _emit_module(self, node: ASTNode, indent_level: int) -> None:
        self._emit_children(node.children, indent_level)

    def _emit_function(self, node: ASTNode, indent_level: int) -> None:
        params = node.params
        param_strs: List[str] = []
        for p in params:
            if p.name in ("self", "cls"):
                # Free Rust functions cannot declare ``self`` receivers; skip them.
                continue
            if p.type_hint:
                param_strs.append(f"{p.name}: {self._map_type(p.type_hint)}")
            else:
                param_strs.append(f"{p.name}: ()")

        declared = self._map_type(node.type_hint)
        inferred = self._infer_return_type_from_body(node.body) if node.body else None
        SignatureBodyMatch.validate(declared, inferred)
        return_type = SignatureBodyMatch.unify(declared, inferred)
        self._current_return_type = return_type

        if self.is_pyo3:
            sig = f"fn {node.name}({', '.join(param_strs)})"
            if not self._is_void(return_type):
                sig += f" -> {return_type}"
            self._write("")
            self._write(f'#[pyfunction(name = "{node.name}")]')
            self._write("#[allow(unused_variables)]", indent_level)
            self._write(f"pub {sig} {{", indent_level)
        elif self.is_c_abi:
            sig = f'extern "C" fn {node.name}({", ".join(param_strs)})'
            if not self._is_void(return_type):
                sig += f" -> {return_type}"
            self._write("")
            self._write("#[no_mangle]", indent_level)
            self._write("#[allow(unused_variables)]", indent_level)
            self._write(f"pub {sig} {{", indent_level)
        else:
            sig = f"fn {node.name}({', '.join(param_strs)})"
            if not self._is_void(return_type):
                sig += f" -> {return_type}"
            self._write("")
            self._write("#[allow(unused_variables)]", indent_level)
            self._write(f"pub {sig} {{", indent_level)
        # Generated bodies often mix i64 (from Python int) with usize (from .len()).
        # Shadow i64 count parameters as usize to avoid E0277/E0308 mismatches.
        for p in params:
            if p.name in ("self", "cls"):
                continue
            if p.type_hint and self._map_type(p.type_hint) == "i64":
                self._write(f"let {p.name} = {p.name} as usize;", indent_level + 1)

        body = node.body
        if body:
            self._emit_children(body, indent_level + 1)
        else:
            self._write(self._default_return_expr(return_type), indent_level + 1)
        self._write("}", indent_level)

    def _is_void(self, type_hint: Optional[str]) -> bool:
        return (type_hint or "").strip() in ("", "void", "None", "()")

    def _infer_return_type_from_body(
        self, body: List[ASTNode]
    ) -> Optional[str]:
        """Infer a Rust return type from the body's ``return`` nodes."""
        types: List[str] = []
        for stmt in body:
            if stmt.kind == "return":
                if stmt.children:
                    types.append(self._expr_type(stmt.children[0]))
                elif stmt.value is not None:
                    types.append(self._expr_type(ASTNode(kind="literal", value=stmt.value)))
        # Drop unit/None values; they should not force a concrete return type.
        concrete = [t for t in types if t not in ("", "void", "None", "()")]
        if not concrete:
            return None
        if "f64" in concrete:
            return "f64"
        if "String" in concrete:
            return "String"
        if all(t == "bool" for t in concrete):
            return "bool"
        # Prefer the first concrete type, falling back to i64.
        return concrete[0] if concrete else "i64"

    def _expr_type(self, node: ASTNode) -> str:
        """Best-effort Rust type for an expression node."""
        if node.type_hint:
            mapped = self._map_type(node.type_hint)
            if not self._is_void(mapped):
                return mapped
        if node.kind == "literal":
            value = node.value
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "i64"
            if isinstance(value, float):
                return "f64"
            if isinstance(value, str):
                return "String"
            if value is None:
                return "()"
        if node.kind == "binary_op":
            left = self._expr_type(node.children[0]) if node.children else "i64"
            right = self._expr_type(node.children[1]) if len(node.children) > 1 else "i64"
            if "f64" in (left, right):
                return "f64"
            if "String" in (left, right):
                return "String"
            return left or "i64"
        if node.kind == "call" and node.name:
            # Heuristic: math helpers returning f64, len returning i64.
            if node.name in ("len",):
                return "i64"
        return "i64"

    def _default_return_expr(self, type_hint: Optional[str]) -> str:
        """Return a value-initialized default expression for a function with no body."""
        t = (type_hint or "").strip()
        if t in ("", "void", "None", "()"):
            return "()"
        if t.startswith("PyResult<") or t.startswith("Result<"):
            return "Ok(Default::default())"
        if t.startswith("Option<"):
            return "None"
        return "Default::default()"

    def _emit_struct(self, node: ASTNode, indent_level: int) -> None:
        self._write("")
        self._write(f"pub struct {node.name} {{", indent_level)
        for field in node.children:
            if field.kind != "field":
                continue
            type_str = self._map_type(field.type_hint) if field.type_hint else "()"
            self._write(f"pub {field.name}: {type_str},", indent_level + 1)
        self._write("}", indent_level)

    def _emit_binding(self, node: ASTNode, indent_level: int) -> None:
        value_str = (
            self._expr(node.children[0]) if node.children else self._literal(node.value)
        )
        if node.type_hint:
            self._write(
                f"let {node.name}: {self._map_type(node.type_hint)} = {value_str};",
                indent_level,
            )
        else:
            self._write(f"let {node.name} = {value_str};", indent_level)

    def _emit_return(self, node: ASTNode, indent_level: int) -> None:
        if self._is_void(self._current_return_type):
            # Void functions must not return a value.
            self._write("return;", indent_level)
            return
        value_node = node.children[0] if node.children else ASTNode(
            kind="literal", value=node.value
        )
        value_str = self._expr(value_node)
        from_type = self._expr_type(value_node)
        value_str = self._cast_return_expression(value_str, from_type, self._current_return_type)
        self._write(f"return {value_str};", indent_level)

    def _cast_return_expression(
        self, value_str: str, from_type: Optional[str], to_type: Optional[str]
    ) -> str:
        """Cast a return expression to the declared return type when safe."""
        if SignatureBodyMatch._is_void(to_type) or from_type == to_type:
            return value_str
        from_type = (from_type or "").strip()
        to_type = (to_type or "").strip()
        if SignatureBodyMatch._is_numeric(from_type) and SignatureBodyMatch._is_numeric(to_type):
            return f"({value_str} as {to_type})"
        return value_str

    def _emit_import(self, node: ASTNode, indent_level: int) -> None:
        self._write(f"use {node.value};", indent_level)

    def _emit_comment(self, node: ASTNode, indent_level: int) -> None:
        for line in str(node.value or "").splitlines():
            self._write(f"// {line}", indent_level)

    def _emit_if(self, node: ASTNode, indent_level: int) -> None:
        if not node.children:
            return
        condition = self._expr(node.children[0])
        self._write(f"if {condition} {{", indent_level)
        if len(node.children) > 1:
            self._emit_children(node.children[1].children, indent_level + 1)
        self._write("}", indent_level)
        if len(node.children) > 2:
            self._write("else {", indent_level)
            self._emit_children(node.children[2].children, indent_level + 1)
            self._write("}", indent_level)

    def _emit_for(self, node: ASTNode, indent_level: int) -> None:
        """Emit a ``for`` loop without unnecessary parentheses around the target."""
        if len(node.children) < 2:
            return
        iter_node = node.children[0]
        body = node.children[1]
        var = node.name or "i"

        if iter_node.kind == "call" and iter_node.name == "range":
            args = iter_node.children
            if len(args) == 1:
                start = "0"
                stop = self._expr(args[0])
            elif len(args) >= 2:
                start = self._expr(args[0])
                stop = self._expr(args[1])
            else:
                start = "0"
                stop = "0"
            iter_expr = f"{start}..{stop}"
        else:
            iter_expr = self._expr(iter_node)

        target = var
        self._write(f"for {target} in {iter_expr} {{", indent_level)
        self._emit_children(body.children, indent_level + 1)
        self._write("}", indent_level)

    def _emit_gil_release(self, node: ASTNode, indent_level: int) -> None:
        """Emit ``py.allow_threads(|| { ... })`` around a block of statements."""
        self._write("py.allow_threads(|| {", indent_level)
        self._emit_children(node.children, indent_level + 1)
        self._write("})", indent_level)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_type(self, type_hint: Optional[str]) -> str:
        if not type_hint:
            return "()"
        hint = type_hint.strip()

        # Preserve explicit PyO3 / numpy-rust signatures.
        if hint.startswith("PyResult<"):
            inner = hint[9:].strip()
            if inner.endswith(">"):
                inner = inner[:-1].strip()
            return f"PyResult<{self._map_type(inner)}>"

        if hint in ("Python", "Py"):
            return "Python"

        # PyO3 functions can accept Python lists as Vec<T>; a bare "pointer" is too
        # ambiguous to map to a C pointer in a PyO3 context, so default to f64 vector.
        if self.is_pyo3 and hint in ("pointer", "vec", "vector"):
            return "Vec<f64>"

        if re.match(r"^&?PyArray\d*<", hint) or hint in (
            "np.ndarray",
            "numpy.ndarray",
            "ndarray",
        ):
            # Default 2-D f64 numpy array for unqualified ndarray annotations.
            if re.search(r"<[^>]+>", hint):
                inner_match = re.search(r"<([^>]+)>", hint)
                inner = inner_match.group(1).strip() if inner_match else "f64"  # type: ignore[union-attr]
                return f"&PyArray2<{inner}>"
            return "&PyArray2<f64>"

        type_map = {
            "int": "i64",
            "float": "f64",
            "double": "f64",
            "str": "String",
            "string": "String",
            "String": "String",
            "bool": "bool",
            "void": "()",
            "None": "()",
            "list": "Vec<()>",
            "dict": "std::collections::HashMap<String, ()>",
            "char": "char",
            "unsigned int": "u64",
            "long": "i64",
            "short": "i16",
            "size_t": "usize",
            "usize": "usize",
            "i64": "i64",
            "int64": "i64",
            "f64": "f64",
            "float64": "f64",
            "u64": "u64",
            "i32": "i32",
            "int32": "i32",
            "f32": "f32",
            "float32": "f32",
            "u32": "u32",
        }

        if hint.startswith("list[") and hint.endswith("]"):
            inner = hint[5:-1]
            return f"Vec<{self._map_type(inner)}>"
        if hint.startswith("dict[") and hint.endswith("]"):
            inner = hint[5:-1]
            if "," in inner:
                k, v = inner.split(",", 1)
                return f"std::collections::HashMap<{self._map_type(k.strip())}, {self._map_type(v.strip())}>"
            return f"std::collections::HashMap<String, {self._map_type(inner)}>"
        return type_map.get(hint, hint)

    def _bool_literal(self, value: bool) -> str:
        return "true" if value else "false"

    def _none_literal(self) -> str:
        return "()"

    def _list_literal(self, children: List[ASTNode]) -> str:
        items = ", ".join(self._expr(c) for c in children)
        return f"vec![{items}]"

    def _dict_literal(self, pairs: List[ASTNode]) -> str:
        entries = []
        for pair in pairs:
            k, v = pair.children
            entries.append(f"({self._expr(k)}, {self._expr(v)})")
        if not entries:
            return "std::collections::HashMap::new()"
        return f"std::collections::HashMap::from([{', '.join(entries)}])"

    def _emit_expression_to_string(self, node: ASTNode) -> str:
        if node.kind == "binary_op" and node.name == "**":
            base, exp = node.children
            return f"{self._expr(base)}.pow({self._expr(exp)} as u32)"
        return super()._emit_expression_to_string(node)

    # ------------------------------------------------------------------
    # PyO3 / module helpers
    # ------------------------------------------------------------------

    def _is_pyo3_spec(self, spec: EngineSpec) -> bool:
        if spec.metadata.get("pyo3"):
            return True
        binding = (spec.metadata.get("boundary_type") or "").lower().replace("-", "_")
        if binding in ("pyo3", "pyo3_maturin", "maturin"):
            return True
        for node in self._all_nodes(spec.root):
            if node.type_hint and any(
                pat in node.type_hint
                for pat in ("PyArray", "Python", "PyResult", "Py<", "&Py")
            ):
                return True
        return False

    def _is_c_abi_spec(self, spec: EngineSpec) -> bool:
        if spec.metadata.get("c_abi"):
            return True
        binding = (spec.metadata.get("boundary_type") or "").lower().replace("-", "_")
        if binding in ("c_abi", "cabi", "c", "raw_c", "ctypes", "cffi"):
            return True
        return False

    def _uses_numpy(self, spec: EngineSpec) -> bool:
        if spec.metadata.get("numpy"):
            return True
        for node in self._all_nodes(spec.root):
            if node.type_hint and (
                "PyArray" in node.type_hint
                or node.type_hint in ("np.ndarray", "numpy.ndarray", "ndarray")
            ):
                return True
        return False

    def _uses_rayon(self, spec: EngineSpec) -> bool:
        if spec.metadata.get("rayon"):
            return True
        for node in self._all_nodes(spec.root):
            value = str(node.value or "")
            name = str(node.name or "")
            if "par_iter" in value or "par_iter" in name:
                return True
        return False

    def _all_nodes(self, root: ASTNode) -> List[ASTNode]:
        nodes: List[ASTNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    def _module_name_from_path(self, path: str) -> str:
        name = Path(path).stem
        return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_") or "module"

    def _emit_module_file(self, mod_root: ASTNode, mod_name: str) -> str:
        """Render a submodule file (e.g. ``src/ops.rs``)."""
        old_lines = self._lines
        self._lines = []
        if self.is_pyo3:
            self._write("use pyo3::prelude::*;")
            if self.uses_numpy:
                self._write("use numpy::PyArray2;")
            self._write("")
        self._emit_children(mod_root.children, 0)
        source = "\n".join(self._lines) + "\n"
        self._lines = old_lines
        return source

    def _emit_pymodule_block(
        self,
        spec: EngineSpec,
        root_functions: List[ASTNode],
        module_registrations: List[Tuple[str, str]],
    ) -> None:
        """Emit a ``#[pymodule]`` block that registers all public functions."""
        module_name = self._sanitize_module_name(spec.name)
        self._write("")
        self._write(f"#[pymodule]")
        self._write(f"fn {module_name}(_py: Python, m: &PyModule) -> PyResult<()> {{")
        for func in root_functions:
            if func.name:
                self._write(f"    m.add_wrapped(wrap_pyfunction!({func.name}))?;")
        for mod_name, func_name in module_registrations:
            self._write(
                f"    m.add_wrapped(wrap_pyfunction!({mod_name}::{func_name}))?;"
            )
        self._write("    Ok(())")
        self._write("}")

    def _sanitize_module_name(self, name: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        if not sanitized or sanitized[0].isdigit():
            sanitized = "aero_forge_native_" + sanitized
        return sanitized


BUILD_RS_TEMPLATE = """fn main() {
    // Aero-Forge build script configuration.
    println!("cargo:rerun-if-changed=build.rs");
}
"""


class RustEmitterPlugin(PolyglotEmitterPlugin):
    """Polyglot plugin adapter for the Rust emitter."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="rust",
            supported_boundaries={
                BoundaryContract.C_ABI,
                BoundaryContract.PYO3_MATURIN,
                BoundaryContract.WASM_WASI,
            },
            toolchains=["rustc", "cargo", "maturin"],
            file_extensions=[".rs"],
            supports_zero_copy=True,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: dict,
        boundary_contracts: List[dict],
    ) -> List[CodeArtifact]:
        spec = _rust_engine_spec_from_node_spec(node_id, node_spec, boundary_contracts)
        emitter = RustEmitter()
        source = emitter.emit(spec)
        artifacts = [CodeArtifact(file_path="src/lib.rs", content=source, language="rust")]
        for mod_path, mod_source in emitter.module_sources.items():
            artifacts.append(
                CodeArtifact(file_path=mod_path, content=mod_source, language="rust")
            )
        # Always emit a valid build.rs so Cargo never sees an empty/placeholder script.
        artifacts.append(
            CodeArtifact(file_path="build.rs", content=BUILD_RS_TEMPLATE, language="rust")
        )
        # Persist the PyO3/C-ABI decision so emit_build_manifest can add the right deps.
        self._last_is_pyo3 = emitter.is_pyo3
        self._last_is_c_abi = emitter.is_c_abi
        return artifacts

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        crate = node_id or "rust_project"
        extra_deps: List[str] = []
        if getattr(self, "_last_is_pyo3", False):
            extra_deps.append('pyo3 = { version = "0.20.3", features = ["extension-module"] }')
        if getattr(self, "_last_is_c_abi", False):
            extra_deps.append('rayon = "1.10"')
        deps = "\n".join(extra_deps + [f'{d} = "0.1"' for d in dependencies])
        crate_type = '["cdylib", "rlib"]' if getattr(self, "_last_is_pyo3", False) else '["cdylib"]'
        content = (
            "[package]\n"
            f'name = "{crate}"\n'
            'version = "0.1.0"\n'
            'edition = "2021"\n'
            'build = "build.rs"\n\n'
            "[lib]\n"
            'name = "' + crate.replace("-", "_") + '"\n'
            f'crate-type = {crate_type}\n\n'
            "[dependencies]\n"
            f"{deps}\n"
        )
        return CodeArtifact(file_path="Cargo.toml", content=content, language="toml")


def _rust_engine_spec_from_node_spec(
    node_id: str,
    node_spec: dict,
    boundary_contracts: Optional[List[dict]] = None,
) -> EngineSpec:
    """Best-effort conversion of a plugin node spec to an EngineSpec."""
    spec = node_spec.get("spec")
    if not isinstance(spec, EngineSpec):
        if "source" in node_spec:
            name = node_spec.get("name") or node_id or "module"
            spec = EngineSpec(name=name, root=module(children=[function(name, body=[])]))
        elif "root" in node_spec:
            spec = EngineSpec(
                name=node_spec.get("name", node_id or "module"),
                root=node_spec["root"],
            )
        else:
            spec = EngineSpec(name=node_id or "module", root=module(children=[]))

    if boundary_contracts:
        _inject_boundary_metadata(spec.metadata, boundary_contracts)
        source_contracts = [
            c for c in boundary_contracts if c.get("source") == node_id and c.get("symbol")
        ]
        if source_contracts:
            spec = _align_rust_spec_to_source_contracts(spec, source_contracts, node_id)
    return spec


def _align_rust_spec_to_source_contracts(
    spec: EngineSpec,
    source_contracts: List[dict],
    node_id: str,
) -> EngineSpec:
    """Return a spec whose top-level functions match the contracted source symbols."""
    existing_funcs = [c for c in spec.root.children if c.kind == "function"]
    new_funcs: List[ASTNode] = []
    for i, contract in enumerate(source_contracts):
        sym = str(contract.get("symbol") or node_id)
        args = list(contract.get("args") or [])
        ret = contract.get("return_type") or ""
        if i < len(existing_funcs):
            existing = existing_funcs[i]
            base_params = existing.params
            body = existing.body
            name = sym or existing.name or node_id
            params = [
                param(
                    base_params[j].name if j < len(base_params) else f"arg_{j}",
                    args[j] if j < len(args) else (base_params[j].type_hint if j < len(base_params) else None),
                )
                for j in range(max(len(args), len(base_params)))
            ]
            ret = ret or existing.type_hint
        else:
            name = sym or node_id
            params = [param(f"arg_{j}", args[j]) for j in range(len(args))]
            body = []
        new_funcs.append(function(name, params=params, return_type=ret or None, body=body))
    if not new_funcs:
        return spec
    return EngineSpec(name=spec.name, root=module(children=new_funcs), metadata=spec.metadata)


def _inject_boundary_metadata(
    metadata: dict, boundary_contracts: List[dict]
) -> None:
    """Promote the dominant boundary contract into spec metadata for the emitter."""
    for contract in boundary_contracts:
        boundary = str(contract.get("boundary_type") or contract.get("boundary") or "c_abi").lower().replace("-", "_")
        metadata["boundary_type"] = boundary
        metadata["pyo3"] = boundary in ("pyo3", "pyo3_maturin", "maturin")
        metadata["c_abi"] = boundary in ("c_abi", "cabi", "c", "raw_c", "ctypes", "cffi")
        if boundary in ("pyo3", "pyo3_maturin", "maturin"):
            break


EmitterRegistry.get_instance().register(RustEmitterPlugin())
