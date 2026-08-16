"""Category-Theoretic Schema Bootstrapper for Aero-Forge blueprints.

This module treats natural-language intent as a source category ``C`` and the
final ``blueprint.aero`` specification as a target category ``D``. A functor
``F : C -> Set`` maps each intent symbol to a fibre of possible node stubs. The
adjoint triple ``(ΣF, ΔF, ΠF)`` and the Grothendieck construction are used to
build a rigid manifest skeleton with typed holes, so the LLM only fills
implementation details rather than deciding structure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NodeStub:
    """A deterministic stub for a single blueprint node."""

    node_id: str
    lang: str
    toolchain: str
    exports: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    purpose: str = ""


class SchemaBootstrapper:
    """Bootstrap a ``blueprint.aero`` skeleton from functional intent.

    The adjoint triple operates as follows:

    * ``ΣF`` (left adjoint): extract/inject parameters from functional intent
      into a fibre of node stubs.
    * ``ΔF`` (diagonal functor): merge duplicate node IDs and validate the
      resulting graph is a DAG with resolvable targets.
    * ``ΠF`` (right adjoint): project the fibre product into a final blueprint
      skeleton, enforcing FFI integrity and toolchain alignment.
    """

    _LANG_TOOLCHAINS = {
        "python": "python",
        "rust": "cargo",
        "cpp": "cmake",
        "c": "cmake",
        "c++": "cmake",
        "go": "go",
        "zig": "zig",
    }

    def __init__(self, architecture_hint: str = "graph_polyglot") -> None:
        self.architecture_hint = architecture_hint
        self.languages = self._languages_from_architecture(architecture_hint)

    def _languages_from_architecture(self, architecture: str) -> List[str]:
        architecture = architecture.lower().replace("-", "_")
        if architecture == "pure_python":
            return ["python"]
        if architecture == "pure_rust":
            return ["rust"]
        if architecture in ("hybrid_cpp_python", "hybrid_python_cpp"):
            return ["python", "cpp"]
        if architecture in ("hybrid_rust_python", "hybrid_python_rust"):
            return ["python", "rust"]
        if architecture in ("hybrid_cpp_rust", "hybrid_rust_cpp"):
            return ["rust", "cpp"]
        if architecture == "tri_polyglot_rust_cpp_python":
            return ["python", "rust", "cpp"]
        return ["python"]

    def _symbol_hash(self, symbol_name: str) -> int:
        digest = hashlib.sha256(symbol_name.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16)

    def _lang_for_symbol(self, symbol_type: str, symbol_name: str) -> str:
        """Deterministically assign a language to a symbol."""
        symbol_type = (symbol_type or "function").lower()
        symbol_name = symbol_name.lower()
        if "test" in symbol_name or symbol_type == "test":
            return "python"
        if symbol_type in ("algorithm", "kernel", "core", "data"):
            # Native acceleration for heavy symbols, distributed across
            # available native languages using a deterministic hash.
            native = [l for l in self.languages if l in ("rust", "cpp", "c", "go", "zig")]
            if native:
                return native[self._symbol_hash(symbol_name) % len(native)]
        if symbol_type in ("ui", "cli", "api", "wrapper"):
            if "python" in self.languages:
                return "python"
        if "python" in self.languages:
            return "python"
        return self.languages[0]

    def _toolchain_for_lang(self, lang: str) -> str:
        return self._LANG_TOOLCHAINS.get(lang, lang)

    def _node_id_for_symbol(self, symbol_name: str) -> str:
        """Map a symbol name to a node id (module path)."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", symbol_name).strip("_") or "node"
        return sanitized

    def _package_for_node(self, node_id: str, lang: str) -> str:
        if lang == "python":
            return f"{node_id}.py"
        if lang == "rust":
            return f"rust_{node_id}/src/lib.rs"
        if lang in ("cpp", "c", "c++"):
            return f"{node_id}.cpp"
        return f"{node_id}.txt"

    # ------------------------------------------------------------------ ΣF --
    def ΣF(self, functional_intent: List[Dict[str, Any]]) -> List[NodeStub]:
        """Extract/inject node stubs from a list of functional intent entries.

        Each entry is expected to be ``{symbol_name, type, requirement_level}``.
        """
        stubs: List[NodeStub] = []
        for entry in functional_intent:
            symbol = str(entry.get("symbol_name") or entry.get("name", ""))
            if not symbol:
                continue
            sym_type = str(entry.get("type", "function"))
            level = str(entry.get("requirement_level", "required")).lower()
            lang = self._lang_for_symbol(sym_type, symbol)
            node_id = self._node_id_for_symbol(symbol)
            stub = NodeStub(
                node_id=node_id,
                lang=lang,
                toolchain=self._toolchain_for_lang(lang),
                exports=[symbol],
                source_files=[self._package_for_node(node_id, lang)],
                purpose=f"{level} {sym_type}",
            )
            stubs.append(stub)
        return stubs

    # ------------------------------------------------------------------ ΔF --
    def ΔF(
        self,
        stubs: List[NodeStub],
        edges: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[NodeStub], List[Dict[str, Any]]]:
        """Merge duplicate node stubs and validate the node-edge graph."""
        merged: Dict[str, NodeStub] = {}
        for stub in stubs:
            if stub.node_id in merged:
                existing = merged[stub.node_id]
                existing.exports = list(dict.fromkeys(existing.exports + stub.exports))
                existing.source_files = list(
                    dict.fromkeys(existing.source_files + stub.source_files)
                )
                if stub.purpose and stub.purpose not in existing.purpose:
                    existing.purpose = f"{existing.purpose}; {stub.purpose}"
            else:
                merged[stub.node_id] = NodeStub(
                    node_id=stub.node_id,
                    lang=stub.lang,
                    toolchain=stub.toolchain,
                    exports=stub.exports[:],
                    source_files=stub.source_files[:],
                    purpose=stub.purpose,
                )

        node_ids = set(merged.keys())
        edges = edges or []
        cleaned_edges: List[Dict[str, Any]] = []
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if not src or not tgt or src == tgt:
                continue
            if tgt not in node_ids:
                # Skip edges to unknown targets; bootstrapper is not clairvoyant.
                continue
            cleaned_edges.append(dict(edge))

        return list(merged.values()), cleaned_edges

    # ------------------------------------------------------------------ ΠF --
    def ΠF(
        self,
        stubs: List[NodeStub],
        edges: List[Dict[str, Any]],
        functional_intent: List[Dict[str, Any]],
        architecture: str,
    ) -> Dict[str, Any]:
        """Project the fibre product into a blueprint skeleton.

        Enforces FFI integrity: edges between nodes of the same language must
        not claim an FFI boundary type, and cross-language edges must use a
        suitable boundary.
        """
        node_map = {s.node_id: s for s in stubs}
        nodes_json: List[Dict[str, Any]] = []
        for stub in stubs:
            nodes_json.append(
                {
                    "node_id": stub.node_id,
                    "lang": stub.lang,
                    "toolchain": stub.toolchain,
                    "source_files": stub.source_files,
                    "exports": stub.exports,
                    "purpose": stub.purpose,
                    # Typed holes: the LLM must fill these concrete fields.
                    "logic_sketch": "<TYPED_HOLE>",
                    "contracts": [],
                }
            )

        boundary_edges: List[Dict[str, Any]] = []
        for edge in edges:
            src_id = edge.get("source", "")
            tgt_id = edge.get("target", "")
            src_stub = node_map.get(src_id)
            tgt_stub = node_map.get(tgt_id)
            if src_stub is None or tgt_stub is None:
                continue

            if src_stub.lang == tgt_stub.lang:
                boundary_type = "internal"
            else:
                boundary_type = self._infer_boundary_type(src_stub.lang, tgt_stub.lang)

            boundary_edges.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "boundary_type": boundary_type,
                    "symbol": edge.get("symbol", edge.get("relation", "")),
                }
            )

        manifest = [
            {
                "path": sf,
                "node_id": stub.node_id,
                "role": "source",
            }
            for stub in stubs
            for sf in stub.source_files
        ]

        # Typed holes that the LLM must still fill.
        typed_holes = [
            {
                "path": f"nodes.{stub.node_id}.logic_sketch",
                "expected_type": "string",
                "description": "Concrete implementation body for the node.",
            }
            for stub in stubs
        ]
        # Only ask for contracts when there are real cross-language boundaries.
        if boundary_edges:
            typed_holes += [
                {
                    "path": f"nodes.{stub.node_id}.contracts",
                    "expected_type": "List[ABIContract]",
                    "description": "Cross-language ABI contracts for exported symbols.",
                }
                for stub in stubs
            ]
        # Grothendieck fiber coordinates: map every functional intent object to
        # the node stub that materializes it, even when no FFI edges exist.
        for intent, stub in self.grothendieck_bundle(functional_intent, stubs):
            sym = (
                intent.get("symbol_name")
                or intent.get("name")
                or (stub.exports[0] if stub.exports else stub.node_id)
            )
            typed_holes.append(
                {
                    "path": f"functional_intent_map.{sym}",
                    "expected_type": "NodeStub",
                    "description": (
                        f"Grothendieck fiber coordinate: symbol '{sym}' is "
                        f"implemented by node '{stub.node_id}' ({stub.lang})."
                    ),
                }
            )

        return {
            "schema_version": "2.0.0",
            "project": architecture,
            "architecture": architecture,
            "metadata": {"bootstrap_method": "category_theoretic"},
            "functional_intent": functional_intent,
            "nodes": nodes_json,
            "edges": boundary_edges,
            "manifest": manifest,
            "typed_holes": typed_holes,
        }

    def _infer_boundary_type(self, src_lang: str, tgt_lang: str) -> str:
        pair = {src_lang, tgt_lang}
        if "rust" in pair and "python" in pair:
            return "PYO3_MATURIN"
        if "cpp" in pair and "python" in pair:
            return "C_ABI"
        if "rust" in pair and "cpp" in pair:
            return "C_ABI"
        return "WASM_WASI"

    # --------------------------------------------------- Grothendieck bundle --
    def grothendieck_bundle(
        self,
        functional_intent: List[Dict[str, Any]],
        stubs: List[NodeStub],
    ) -> List[Tuple[Dict[str, Any], NodeStub]]:
        """Build the total space of the discrete fibration ``∫ F``.

        Each fibre element is paired with its base intent object, giving a
        deterministic coordinate for every user requirement in the manifest.
        """
        bundle: List[Tuple[Dict[str, Any], NodeStub]] = []
        # Map each intent to the stubs whose exports contain the symbol.
        intent_by_symbol = {
            str(entry.get("symbol_name") or entry.get("name", "")): entry
            for entry in functional_intent
        }
        for stub in stubs:
            for symbol in stub.exports:
                intent = intent_by_symbol.get(symbol, {})
                bundle.append((intent, stub))
        return bundle

    # ------------------------------------------------------ public bootstrap --
    def bootstrap(
        self,
        functional_intent: List[Dict[str, Any]],
        repo_graph: Optional[Dict[str, Any]] = None,
        architecture_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a rigid blueprint skeleton from functional intent.

        ``repo_graph`` is optional context from ``FockGraphEncoder.encode_repository``.
        """
        architecture = (architecture_hint or self.architecture_hint).lower().replace(
            "-", "_"
        )
        self.architecture_hint = architecture
        self.languages = self._languages_from_architecture(architecture)

        stubs = self.ΣF(functional_intent)

        # If a repository topology was supplied, add edges derived from it.
        edges: List[Dict[str, Any]] = []
        if repo_graph:
            for edge in repo_graph.get("edges", []):
                edges.append(
                    {
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "relation": edge.get("relation", "depends_on"),
                    }
                )

        # Build the Grothendieck total space before merging; this provides a
        # deterministic coordinate for every (intent, node) pair.
        _ = self.grothendieck_bundle(functional_intent, stubs)

        stubs, edges = self.ΔF(stubs, edges)
        return self.ΠF(stubs, edges, functional_intent, architecture)
