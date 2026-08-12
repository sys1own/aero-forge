"""Logical Firewall for blueprint manifests using W3C SHACL.

Translates a candidate ``blueprint.aero`` manifest into an RDF graph and
validates it against a shapes graph that encodes platform architecture and
safety requirements. The firewall guarantees that the manifest is structurally
sound before any source files are materialised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rdflib import Graph, Literal, Namespace, RDF, RDFS, SH, XSD, URIRef

try:
    from pyshacl import validate
except ImportError:  # pragma: no cover
    validate = None  # type: ignore


AERO = Namespace("http://aero-forge.dev/ontology/")


@dataclass
class FirewallReport:
    """Result of a SHACL validation run."""

    conforms: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""

    def to_llm_feedback(self) -> str:
        if self.conforms:
            return "SHACL validation passed: manifest conforms to platform safety rules."
        lines = ["SHACL validation failed. Violations:"]
        for v in self.violations:
            lines.append(f"- [{v.get('severity', 'violation')}] {v.get('message', '')}")
        return "\n".join(lines)


class LogicalFirewall:
    """SHACL-based logical firewall for Aero-Forge manifests.

    The firewall converts a manifest dictionary into an RDF graph, validates it
    against a built-in SHACL shapes graph, and returns a structured report. The
    LLM is only exposed to the validation feedback and a compact RDF summary,
    never to core source files.
    """

    _ALLOWED_ARCHITECTURES = [
        "pure_python",
        "pure_rust",
        "hybrid_rust_python",
        "hybrid_cpp_python",
        "hybrid_cpp_rust",
        "tri_polyglot_rust_cpp_python",
        "graph_polyglot",
    ]

    _ALLOWED_LANGS = ["python", "rust", "cpp", "c", "go", "zig", "javascript"]

    _LANG_TOOLCHAINS = {
        "python": {"python"},
        "rust": {"cargo", "rustc"},
        "cpp": {"cmake", "clang", "gcc", "g++", "clang++"},
        "c": {"cmake", "clang", "gcc"},
        "go": {"go"},
        "zig": {"zig"},
    }

    _BOUNDARY_LANG_COMPAT = {
        "PYO3_MATURIN": {("python", "rust"), ("rust", "python")},
        "C_ABI": {
            ("python", "cpp"),
            ("cpp", "python"),
            ("rust", "cpp"),
            ("cpp", "rust"),
            ("python", "c"),
            ("c", "python"),
            ("rust", "c"),
            ("c", "rust"),
        },
        "WASM_WASI": set(),
        "INTERNAL": set(),
    }

    def __init__(self, manifest: Optional[Dict[str, Any]] = None) -> None:
        self.manifest = manifest or {}
        self.data_graph = Graph()
        self.shapes_graph = Graph()
        self._build_shapes()

    def _build_shapes(self) -> None:
        """Construct the built-in SHACL shapes graph."""
        g = self.shapes_graph
        g.bind("aero", AERO)
        g.bind("sh", SH)

        # Blueprint shape: architecture must be in the allowed list and at least
        # one node must exist.
        blueprint_shape = URIRef("http://aero-forge.dev/shapes/BlueprintShape")
        g.add((blueprint_shape, RDF.type, SH.NodeShape))
        g.add((blueprint_shape, SH.targetClass, AERO.Blueprint))
        g.add(
            (
                blueprint_shape,
                SH.property,
                URIRef("http://aero-forge.dev/shapes/BlueprintArchitecture"),
            )
        )
        g.add(
            (
                blueprint_shape,
                SH.property,
                URIRef("http://aero-forge.dev/shapes/BlueprintNodeCount"),
            )
        )

        arch_prop = URIRef("http://aero-forge.dev/shapes/BlueprintArchitecture")
        g.add((arch_prop, SH.path, AERO.architecture))
        g.add((arch_prop, SH["in"], self._rdf_list(self._ALLOWED_ARCHITECTURES)))
        g.add(
            (
                arch_prop,
                SH.message,
                Literal("Architecture is not a supported platform architecture."),
            )
        )

        node_count_prop = URIRef("http://aero-forge.dev/shapes/BlueprintNodeCount")
        g.add((node_count_prop, SH.path, AERO.hasNode))
        g.add((node_count_prop, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add(
            (
                node_count_prop,
                SH.message,
                Literal("A blueprint must contain at least one node."),
            )
        )

        # Node shape: lang must be supported and a toolchain must be present.
        node_shape = URIRef("http://aero-forge.dev/shapes/NodeShape")
        g.add((node_shape, RDF.type, SH.NodeShape))
        g.add((node_shape, SH.targetClass, AERO.Node))
        g.add((node_shape, SH.property, URIRef("http://aero-forge.dev/shapes/NodeLang")))
        g.add(
            (node_shape, SH.property, URIRef("http://aero-forge.dev/shapes/NodeToolchain"))
        )
        g.add(
            (node_shape, SH.property, URIRef("http://aero-forge.dev/shapes/NodeExports"))
        )

        node_lang = URIRef("http://aero-forge.dev/shapes/NodeLang")
        g.add((node_lang, SH.path, AERO.lang))
        g.add((node_lang, SH["in"], self._rdf_list(self._ALLOWED_LANGS)))
        g.add((node_lang, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add((node_lang, SH.message, Literal("Node lang is not supported.")))

        node_toolchain = URIRef("http://aero-forge.dev/shapes/NodeToolchain")
        g.add((node_toolchain, SH.path, AERO.toolchain))
        g.add((node_toolchain, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add(
            (
                node_toolchain,
                SH.message,
                Literal("Every node must declare a toolchain."),
            )
        )

        node_exports = URIRef("http://aero-forge.dev/shapes/NodeExports")
        g.add((node_exports, SH.path, AERO.exports))
        g.add((node_exports, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add(
            (
                node_exports,
                SH.message,
                Literal("Every node must export at least one symbol."),
            )
        )

        # Edge shape: boundary_type must be valid for the source/target lang pair.
        edge_shape = URIRef("http://aero-forge.dev/shapes/EdgeShape")
        g.add((edge_shape, RDF.type, SH.NodeShape))
        g.add((edge_shape, SH.targetClass, AERO.Edge))
        g.add((edge_shape, SH.property, URIRef("http://aero-forge.dev/shapes/EdgeBoundary")))

        edge_boundary = URIRef("http://aero-forge.dev/shapes/EdgeBoundary")
        g.add((edge_boundary, SH.path, AERO.boundaryType))
        g.add((edge_boundary, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add(
            (
                edge_boundary,
                SH.message,
                Literal("Every edge must declare a boundary_type."),
            )
        )

        # Functional intent shape: every symbol must be covered.
        intent_shape = URIRef("http://aero-forge.dev/shapes/FunctionalIntentShape")
        g.add((intent_shape, RDF.type, SH.NodeShape))
        g.add((intent_shape, SH.targetClass, AERO.FunctionalIntent))
        g.add(
            (
                intent_shape,
                SH.property,
                URIRef("http://aero-forge.dev/shapes/IntentSymbolName"),
            )
        )

        intent_symbol = URIRef("http://aero-forge.dev/shapes/IntentSymbolName")
        g.add((intent_symbol, SH.path, AERO.symbolName))
        g.add((intent_symbol, SH.minCount, Literal(1, datatype=XSD.integer)))
        g.add(
            (
                intent_symbol,
                SH.message,
                Literal("Every functional intent entry must have a symbol_name."),
            )
        )

    def _rdf_list(self, items: List[str]) -> URIRef:
        """Create an RDF list of literals and return its URI."""
        g = self.shapes_graph
        head = URIRef("urn:aero-forge:rdflist:" + str(hash(tuple(items))))
        current = head
        for i, item in enumerate(items):
            g.add((current, RDF.type, RDF.List))
            g.add((current, RDF.first, Literal(item)))
            if i < len(items) - 1:
                next_node = URIRef(f"urn:aero-forge:rdflist:{hash(tuple(items))}:{i+1}")
                g.add((current, RDF.rest, next_node))
                current = next_node
            else:
                g.add((current, RDF.rest, RDF.nil))
        return head

    def _manifest_to_rdf(self) -> Graph:
        """Convert the manifest dict into an RDF graph."""
        g = Graph()
        g.bind("aero", AERO)
        manifest = self.manifest

        blueprint = AERO["Blueprint"]
        g.add((blueprint, RDF.type, AERO.Blueprint))
        g.add((blueprint, AERO.architecture, Literal(manifest.get("architecture", ""))))
        g.add(
            (
                blueprint,
                AERO.projectName,
                Literal(manifest.get("project", "") or manifest.get("project_name", "")),
            )
        )

        node_map: Dict[str, URIRef] = {}
        for i, node in enumerate(manifest.get("nodes", [])):
            node_id = str(node.get("node_id") or f"node_{i}")
            n = AERO[f"Node/{node_id}"]
            node_map[node_id] = n
            g.add((n, RDF.type, AERO.Node))
            g.add((n, AERO.nodeId, Literal(node_id)))
            g.add((n, AERO.lang, Literal(str(node.get("lang", "")))))
            g.add((n, AERO.toolchain, Literal(str(node.get("toolchain", "")))))
            g.add((blueprint, AERO.hasNode, n))
            for sym in node.get("exports", []):
                g.add((n, AERO.exports, Literal(str(sym))))
            for contract in node.get("contracts", []):
                if isinstance(contract, dict):
                    sym = contract.get("symbol")
                    if sym:
                        g.add((n, AERO.exports, Literal(str(sym))))

        for i, edge in enumerate(manifest.get("edges", [])):
            src_id = str(edge.get("source", ""))
            tgt_id = str(edge.get("target", ""))
            e = AERO[f"Edge/{i}"]
            g.add((e, RDF.type, AERO.Edge))
            g.add((e, AERO.source, Literal(src_id)))
            g.add((e, AERO.target, Literal(tgt_id)))
            g.add((e, AERO.boundaryType, Literal(str(edge.get("boundary_type", "")))))
            g.add((blueprint, AERO.hasEdge, e))

            # Enforce boundary/language compatibility by adding explicit
            # compatibility statements derived from the manifest data.
            src_node = next(
                (n for n in manifest.get("nodes", []) if n.get("node_id") == src_id), {}
            )
            tgt_node = next(
                (n for n in manifest.get("nodes", []) if n.get("node_id") == tgt_id), {}
            )
            src_lang = str(src_node.get("lang", "")).lower()
            tgt_lang = str(tgt_node.get("lang", "")).lower()
            boundary = str(edge.get("boundary_type", "")).upper()
            ok = self._boundary_ok(src_lang, tgt_lang, boundary)
            g.add((e, AERO.boundaryCompatible, Literal(ok, datatype=XSD.boolean)))

        for i, intent in enumerate(manifest.get("functional_intent", [])):
            sym = str(intent.get("symbol_name") or intent.get("name", ""))
            if not sym:
                continue
            fi = AERO[f"FunctionalIntent/{i}"]
            g.add((fi, RDF.type, AERO.FunctionalIntent))
            g.add((fi, AERO.symbolName, Literal(sym)))
            g.add((fi, AERO.requirementLevel, Literal(str(intent.get("requirement_level", "")))))
            g.add((blueprint, AERO.hasFunctionalIntent, fi))
            # Coverage is determined by whether the symbol appears in exports.
            covered = any(
                sym == str(s)
                for node in manifest.get("nodes", [])
                for s in node.get("exports", [])
            ) or any(sym == str(edge.get("symbol", "")) for edge in manifest.get("edges", []))
            g.add((fi, AERO.covered, Literal(covered, datatype=XSD.boolean)))

        return g

    def _boundary_ok(self, src_lang: str, tgt_lang: str, boundary: str) -> bool:
        if src_lang == tgt_lang:
            return boundary in ("", "INTERNAL")
        allowed = self._BOUNDARY_LANG_COMPAT.get(boundary, set())
        return (src_lang, tgt_lang) in allowed

    def _deterministic_checks(self) -> List[Dict[str, Any]]:
        """Run deterministic manifest checks beyond the SHACL shapes graph."""
        manifest = self.manifest
        errors: List[Dict[str, Any]] = []

        node_ids = {str(n.get("node_id", "")): n for n in manifest.get("nodes", [])}
        exports: set = set()
        for node in manifest.get("nodes", []):
            for sym in node.get("exports", []):
                exports.add(str(sym))
            for contract in node.get("contracts", []):
                if isinstance(contract, dict):
                    sym = contract.get("symbol")
                    if sym:
                        exports.add(str(sym))

        # Toolchain/language consistency.
        for node in manifest.get("nodes", []):
            nid = str(node.get("node_id", ""))
            lang = str(node.get("lang", "")).lower()
            toolchain = str(node.get("toolchain", "")).lower()
            valid_toolchains = self._LANG_TOOLCHAINS.get(lang, set())
            if lang and toolchain and toolchain not in valid_toolchains:
                errors.append(
                    {
                        "message": (
                            f"Node '{nid}' uses toolchain '{toolchain}' which is not "
                            f"valid for language '{lang}'."
                        ),
                        "severity": "error",
                    }
                )

        # Edge target/source existence and boundary compatibility.
        for i, edge in enumerate(manifest.get("edges", [])):
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            boundary = str(edge.get("boundary_type", "")).upper()
            if src not in node_ids:
                errors.append(
                    {
                        "message": f"Edge {i} references unknown source node '{src}'.",
                        "severity": "error",
                    }
                )
            if tgt not in node_ids:
                errors.append(
                    {
                        "message": f"Edge {i} references unknown target node '{tgt}'.",
                        "severity": "error",
                    }
                )
            src_lang = str(node_ids.get(src, {}).get("lang", "")).lower()
            tgt_lang = str(node_ids.get(tgt, {}).get("lang", "")).lower()
            if src_lang and tgt_lang and not self._boundary_ok(src_lang, tgt_lang, boundary):
                errors.append(
                    {
                        "message": (
                            f"Edge {i} from '{src}' ({src_lang}) to '{tgt}' "
                            f"({tgt_lang}) has invalid boundary type '{boundary}'."
                        ),
                        "severity": "error",
                    }
                )

        # Functional intent coverage.
        for intent in manifest.get("functional_intent", []):
            sym = str(intent.get("symbol_name") or intent.get("name", ""))
            if sym and sym not in exports:
                errors.append(
                    {
                        "message": (
                            f"Functional intent symbol '{sym}' is not exported by any node."
                        ),
                        "severity": "error",
                    }
                )

        return errors

    def validate(self) -> FirewallReport:
        """Validate the manifest and return a ``FirewallReport``."""
        if validate is None:
            return FirewallReport(
                conforms=False,
                text="pyshacl is not installed; cannot run SHACL validation.",
            )

        data_graph = self._manifest_to_rdf()
        self.data_graph = data_graph

        try:
            conforms, results_graph, results_text = validate(
                data_graph,
                shacl_graph=self.shapes_graph,
                data_graph_format="turtle",
                shacl_graph_format="turtle",
                inference="rdfs",
                debug=False,
            )
        except Exception as exc:
            return FirewallReport(conforms=False, text=str(exc))

        violations = []
        if not conforms and results_graph is not None:
            for result in results_graph.subjects(RDF.type, SH.ValidationResult):
                message = ""
                severity = ""
                for pred, obj in results_graph.predicate_objects(result):
                    if pred == SH.resultMessage:
                        message = str(obj)
                    elif pred == SH.resultSeverity:
                        severity = str(obj).split("#")[-1]
                violations.append(
                    {
                        "message": message,
                        "severity": severity or "sh:Violation",
                        "result": str(result),
                    }
                )

        extra = self._deterministic_checks()
        violations.extend(extra)

        return FirewallReport(
            conforms=conforms and not extra,
            violations=violations,
            text=results_text or "",
        )

    def compact_rdf_summary(self) -> str:
        """Return a compact, LLM-readable Turtle summary of the manifest."""
        g = self._manifest_to_rdf()
        # Serialize with a stable, readable prefix.
        g.bind("aero", AERO)
        return g.serialize(format="turtle") or ""


def validate_manifest(manifest: Dict[str, Any]) -> FirewallReport:
    """Convenience entry point."""
    return LogicalFirewall(manifest).validate()
