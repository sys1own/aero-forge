"""Fock-Space Graph Encoder (FoGE) for repository topology.

FoGE parses a code base with Tree-sitter, extracts inter-module dependencies,
and encodes each dependency as a compact "Prompt-as-Prefix" (PaP) vector using
circular convolution in Fourier space. The resulting token series can be used
by downstream modules (e.g., the Schema Bootstrapper) to seed a blueprint from the
actual shape of an existing repository.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    import tree_sitter_rust
    import tree_sitter_cpp
except ImportError:  # pragma: no cover
    Language = Parser = None  # type: ignore
    tree_sitter_python = tree_sitter_rust = tree_sitter_cpp = None  # type: ignore


@dataclass
class FockEdge:
    """A single topological relation between two modules."""

    source: str
    relation: str
    target: str
    source_file: Optional[str] = None


@dataclass
class FockNode:
    """A module discovered in the repository."""

    name: str
    file_path: Path
    language: str
    exports: List[str] = field(default_factory=list)
    vector: Optional[np.ndarray] = None


class FockGraphEncoder:
    """Encode repository topology as a series of PaP vectors.

    Example:
        encoder = FockGraphEncoder(dim=4096)
        result = encoder.encode_repository(".")
        result["nodes"], result["edges"], result["tokens"]
    """

    def __init__(self, dim: int = 4096, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed
        self._vectors: Dict[str, np.ndarray] = {}
        self._parsers: Dict[str, Parser] = {}
        self._init_parsers()

    def _init_parsers(self) -> None:
        if Parser is None or Language is None:
            return
        try:
            py_lang = Language(tree_sitter_python.language())
            self._parsers[".py"] = Parser(py_lang)
        except Exception:
            pass
        try:
            rs_lang = Language(tree_sitter_rust.language())
            self._parsers[".rs"] = Parser(rs_lang)
        except Exception:
            pass
        try:
            cpp_lang = Language(tree_sitter_cpp.language())
            self._parsers[".cpp"] = Parser(cpp_lang)
            self._parsers[".cc"] = Parser(cpp_lang)
            self._parsers[".cxx"] = Parser(cpp_lang)
            self._parsers[".hpp"] = Parser(cpp_lang)
            self._parsers[".h"] = Parser(cpp_lang)
            self._parsers[".c"] = Parser(cpp_lang)
        except Exception:
            pass

    def _name_seed(self, name: str) -> int:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        return self.seed ^ int(digest, 16)

    def vector(self, name: str) -> np.ndarray:
        """Return a deterministic, unit-norm random vector for ``name``."""
        if name in self._vectors:
            return self._vectors[name]
        rng = np.random.default_rng(self._name_seed(name))
        v = rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        self._vectors[name] = v
        return v

    @staticmethod
    def circular_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution via FFT (real part)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("vectors must have the same dimension")
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

    @staticmethod
    def circular_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular correlation via FFT (real part)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("vectors must have the same dimension")
        return np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real

    def token_for(self, source: str, relation: str, target: str) -> np.ndarray:
        """Encode one dependency as a PaP vector: src ⊗ rel ⊗ dst."""
        src_vec = self.vector(source)
        rel_vec = self.vector(relation)
        dst_vec = self.vector(target)
        return self.circular_convolve(
            self.circular_convolve(src_vec, rel_vec), dst_vec
        )

    def _rel_for_lang(self, lang: str) -> str:
        return {
            ".py": "imports",
            ".rs": "uses",
            ".cpp": "includes",
            ".cc": "includes",
            ".cxx": "includes",
            ".hpp": "includes",
            ".h": "includes",
            ".c": "includes",
        }.get(lang, "depends_on")

    def _extract_python_deps(
        self, file_path: Path, source: str, tree
    ) -> List[str]:
        deps: List[str] = []
        root = tree.root_node

        def visit(node):
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        deps.append(child.text.decode("utf-8"))
            elif node.type == "import_from_statement":
                from_module: Optional[str] = None
                imported: List[str] = []
                for child in node.children:
                    if child.type in ("dotted_name", "relative_import"):
                        from_module = child.text.decode("utf-8")
                    elif child.type == "dotted_name" and from_module is not None:
                        # second dotted_name is the imported symbol
                        imported.append(child.text.decode("utf-8"))
                    elif child.type == "aliased_import":
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                imported.append(sub.text.decode("utf-8"))
                                break
                if from_module:
                    if from_module.startswith("."):
                        from_module = self._resolve_relative(from_module, file_path)
                    for sym in imported:
                        deps.append(f"{from_module}.{sym}")
                    if not imported:
                        deps.append(from_module)
            for child in node.children:
                visit(child)

        visit(root)
        return deps

    def _extract_rust_deps(
        self, file_path: Path, source: str, tree
    ) -> List[str]:
        deps: List[str] = []
        root = tree.root_node

        def visit(node):
            if node.type == "use_declaration":
                for child in node.children:
                    if child.type in ("scoped_identifier", "identifier"):
                        path = child.text.decode("utf-8")
                        if path.startswith("crate::"):
                            path = path[len("crate::"):]
                        deps.append(path)
                        break
            elif node.type == "mod_item":
                for child in node.children:
                    if child.type == "identifier":
                        deps.append(f"mod:{child.text.decode('utf-8')}")
                        break
            for child in node.children:
                visit(child)

        visit(root)
        return deps

    def _extract_cpp_deps(
        self, file_path: Path, source: str, tree
    ) -> List[str]:
        deps: List[str] = []
        root = tree.root_node

        def visit(node):
            if node.type == "preproc_include":
                path_node = None
                for child in node.children:
                    if child.type in ("string_literal", "system_lib_string"):
                        path_node = child
                        break
                if path_node:
                    raw = path_node.text.decode("utf-8")
                    path = raw.strip('"').strip("<>").strip("<>")
                    # Ignore standard/system headers without a local slash.
                    if "/" in path or path.endswith((".h", ".hpp")):
                        deps.append(path)
            for child in node.children:
                visit(child)

        visit(root)
        return deps

    def _resolve_relative(self, dotted: str, file_path: Path) -> str:
        """Resolve a Python relative import to a dotted module name."""
        parts = file_path.parent.relative_to(file_path.anchor).parts
        if parts and parts[0] == "/":
            parts = parts[1:]
        dots = 0
        while dotted.startswith("."):
            dots += 1
            dotted = dotted[1:]
        if dots > len(parts):
            return dotted
        base = parts[: len(parts) - dots + 1] if dots > 0 else parts
        remainder = dotted.split(".") if dotted else []
        return ".".join(base + tuple(remainder))

    def _module_name_for_file(self, file_path: Path, root: Path) -> str:
        """Derive a module identifier from a file path under ``root``."""
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        if parts[0] == "src":
            parts = parts[1:]
        name = "/".join(parts)
        for ext in (
            ".py",
            ".rs",
            ".cpp",
            ".cc",
            ".cxx",
            ".hpp",
            ".h",
            ".c",
        ):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return name.replace("/", ".").strip(".") or file_path.stem

    def _extract_exports(
        self, file_path: Path, language: str, source: str, tree
    ) -> List[str]:
        """Best-effort extraction of top-level exported symbols."""
        exports: List[str] = []
        root = tree.root_node

        def visit(node):
            if language == ".py":
                if node.type in ("function_definition", "class_definition"):
                    for child in node.children:
                        if child.type == "identifier":
                            exports.append(child.text.decode("utf-8"))
                            break
            elif language == ".rs":
                if node.type in ("function_item", "struct_item", "impl_item"):
                    for child in node.children:
                        if child.type == "identifier":
                            exports.append(child.text.decode("utf-8"))
                            break
            elif language in (".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"):
                if node.type in (
                    "function_definition",
                    "function_declaration",
                    "class_specifier",
                ):
                    for child in node.children:
                        if child.type == "identifier":
                            name = child.text.decode("utf-8")
                            if name:
                                exports.append(name)
                            break
            for child in node.children:
                visit(child)

        visit(root)
        # Deduplicate while preserving order.
        seen: set = set()
        out: List[str] = []
        for e in exports:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def regenerate_pap_tokens(
        self,
        root: Path | str,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """Force regeneration of cached PaP vectors and re-encode ``root``."""
        self._vectors = {}
        return self.encode_repository(root, extensions=extensions)

    def encode_repository(
        self,
        root: Path | str,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, any]:
        """Walk ``root`` and return nodes, edges, and PaP tokens."""
        root = Path(root).resolve()
        if extensions is None:
            extensions = (
                ".py",
                ".rs",
                ".cpp",
                ".cc",
                ".cxx",
                ".hpp",
                ".h",
                ".c",
            )

        nodes: Dict[str, FockNode] = {}
        edges: List[FockEdge] = []
        tokens: List[np.ndarray] = []

        for file_path in root.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix not in extensions:
                continue
            # Skip hidden dirs and build artifacts.
            if any(p.startswith(".") for p in file_path.relative_to(root).parts):
                continue
            if "target" in file_path.parts or "__pycache__" in file_path.parts:
                continue

            parser = self._parsers.get(file_path.suffix)
            if parser is None:
                continue

            try:
                source = file_path.read_bytes()
                tree = parser.parse(source)
            except Exception:
                continue

            language = file_path.suffix
            module = self._module_name_for_file(file_path, root)
            exports = self._extract_exports(file_path, language, source, tree)

            if module not in nodes:
                nodes[module] = FockNode(
                    name=module,
                    file_path=file_path,
                    language=language,
                    exports=exports,
                    vector=self.vector(module),
                )
            else:
                nodes[module].exports.extend(exports)

            relation = self._rel_for_lang(language)
            if language == ".py":
                deps = self._extract_python_deps(file_path, source, tree)
            elif language == ".rs":
                deps = self._extract_rust_deps(file_path, source, tree)
            elif language in (".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"):
                deps = self._extract_cpp_deps(file_path, source, tree)
            else:
                deps = []

            for dep in deps:
                if dep == module or not dep:
                    continue
                edge = FockEdge(
                    source=module,
                    relation=relation,
                    target=dep,
                    source_file=str(file_path.relative_to(root)),
                )
                edges.append(edge)
                tokens.append(self.token_for(module, relation, dep))

        return {
            "nodes": {
                name: {
                    "name": node.name,
                    "file_path": str(node.file_path.relative_to(root)),
                    "language": node.language,
                    "exports": node.exports,
                    "vector": node.vector.tolist() if node.vector is not None else None,
                }
                for name, node in nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "relation": e.relation,
                    "target": e.target,
                    "source_file": e.source_file,
                }
                for e in edges
            ],
            "tokens": [t.tolist() for t in tokens],
            "dim": self.dim,
        }

    def similarity(self, token: np.ndarray, query: np.ndarray) -> float:
        """Cosine similarity between two PaP vectors."""
        token = np.asarray(token, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64)
        denom = np.linalg.norm(token) * np.linalg.norm(query)
        if denom == 0:
            return 0.0
        return float(np.dot(token, query) / denom)
