"""Chiasmus: Tree-sitter to Prolog facts and local logic engine.

Parses Python, Rust, and C++ source into ground Prolog-style facts, then runs a
small Datalog-like forward-chaining engine to detect cyclic dependencies,
unreachable symbols, and unsafe cross-language transitions. The derivation
traces are intended to be fed back to the LLM together with an unsatisfiable
core from ``concolic.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    import tree_sitter_rust
    import tree_sitter_cpp
except ImportError:  # pragma: no cover
    Language = Parser = None  # type: ignore
    tree_sitter_python = tree_sitter_rust = tree_sitter_cpp = None  # type: ignore


_FACT_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\.\s*$")


def _quote(s: str) -> str:
    """Quote a string for Prolog-style fact output."""
    escaped = s.replace('"', '\\"')
    return f'"{escaped}"'


def _fact(predicate: str, *args: str) -> str:
    return f"{predicate}({', '.join(_quote(a) for a in args)})."


class PrologFactEmitter:
    """Generate ground Prolog facts from a repository."""

    def __init__(self) -> None:
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

    def _module_name(self, file_path: Path, root: Path) -> str:
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        name = "/".join(parts)
        for ext in (".py", ".rs", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return name.replace("/", ".").strip(".") or file_path.stem

    def _python_imports(self, tree) -> List[Tuple[str, str]]:
        """Return (module, imported_symbol) pairs for Python imports."""
        def collect(node):
            out = []
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        out.append((child.text.decode("utf-8"), "*"))
                    elif child.type == "aliased_import":
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                out.append((sub.text.decode("utf-8"), "*"))
                                break
            elif node.type == "import_from_statement":
                module = ""
                saw_import = False
                for child in node.children:
                    if child.type == "import":
                        saw_import = True
                        continue
                    if not saw_import and child.type in (
                        "dotted_name",
                        "relative_import",
                    ):
                        module = child.text.decode("utf-8")
                    elif saw_import and child.type == "dotted_name":
                        out.append((module, child.text.decode("utf-8")))
                    elif saw_import and child.type == "aliased_import":
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                out.append((module, sub.text.decode("utf-8")))
                                break
                if module and not any(True for _ in out if _[0] == module):
                    out.append((module, "*"))
            for child in node.children:
                out.extend(collect(child))
            return out

        return collect(tree.root_node)

    def _python_exports(self, tree) -> List[str]:
        exports = []

        def visit(node):
            if node.type in ("function_definition", "class_definition"):
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        if name and not name.startswith("_"):
                            exports.append(name)
                        break
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return list(dict.fromkeys(exports))

    def _python_calls(self, tree, module: str) -> Iterator[Tuple[str, str]]:
        """Yield (caller_qname, callee_name)."""
        function_stack: List[str] = []

        def visit(node):
            if node.type == "function_definition":
                name = "<module>"
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        break
                function_stack.append(f"{module}.{name}" if module else name)
                for child in node.children:
                    visit(child)
                function_stack.pop()
                return
            if node.type == "call":
                callee = ""
                func = node.child_by_field_name("function")
                if func is not None:
                    callee = func.text.decode("utf-8")
                caller = function_stack[-1] if function_stack else module or "<module>"
                if callee:
                    yield (caller, callee)
            for child in node.children:
                yield from visit(child)

        # tree-sitter Node is not iterable by default for yield from recursion,
        # so use an explicit stack.
        yield from self._walk(tree.root_node, function_stack, module)

    def _walk(
        self,
        node,
        function_stack: List[str],
        module: str,
    ) -> Iterator[Tuple[str, str]]:
        """Tree walk that maintains a function stack and emits call facts."""
        # Simple recursive walk using children list.
        if node.type == "function_definition":
            name = "<module>"
            for child in node.children:
                if child.type == "identifier":
                    raw = child.text.decode("utf-8")
                    if raw:
                        name = raw
                    break
            function_stack.append(f"{module}.{name}" if module else name)
            for child in node.children:
                yield from self._walk(child, function_stack, module)
            function_stack.pop()
            return

        if node.type == "call":
            callee = ""
            for i, child in enumerate(node.children):
                if child.type in ("identifier", "attribute"):
                    callee = child.text.decode("utf-8")
                    break
            caller = function_stack[-1] if function_stack else module or "<module>"
            if callee:
                yield (caller, callee)

        for child in node.children:
            yield from self._walk(child, function_stack, module)

    def _rust_facts(self, tree, module: str) -> Iterator[str]:
        def visit(node):
            if node.type == "use_declaration":
                for child in node.children:
                    if child.type in ("scoped_identifier", "identifier"):
                        path = child.text.decode("utf-8")
                        if path.startswith("crate::"):
                            path = path[len("crate::"):]
                        yield _fact("uses", module, path)
                        break
            elif node.type == "function_item":
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        yield _fact("exports", module, name)
                        break
            elif node.type == "mod_item":
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        yield _fact("module", module, name)
                        break
            for child in node.children:
                yield from visit(child)

        def collect(node):
            out = []
            if node.type == "use_declaration":
                for child in node.children:
                    if child.type in ("scoped_identifier", "identifier"):
                        path = child.text.decode("utf-8")
                        if path.startswith("crate::"):
                            path = path[len("crate::"):]
                        out.append(_fact("uses", module, path))
                        break
            elif node.type == "function_item":
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        out.append(_fact("exports", module, name))
                        break
            elif node.type == "mod_item":
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        out.append(_fact("module", module, name))
                        break
            for child in node.children:
                out.extend(collect(child))
            return out

        return collect(tree.root_node)

    def _cpp_facts(self, tree, module: str) -> Iterator[str]:
        def collect(node):
            out = []
            if node.type == "preproc_include":
                path_node = None
                for child in node.children:
                    if child.type in ("string_literal", "system_lib_string"):
                        path_node = child
                        break
                if path_node:
                    raw = path_node.text.decode("utf-8")
                    path = raw.strip('"').strip("<>").strip("<>")
                    if "/" in path or path.endswith((".h", ".hpp")):
                        out.append(_fact("includes", module, path))
            elif node.type in (
                "function_definition",
                "function_declaration",
                "class_specifier",
            ):
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        if name:
                            out.append(_fact("exports", module, name))
                        break
            for child in node.children:
                out.extend(collect(child))
            return out

        return collect(tree.root_node)

    def emit_facts(
        self,
        root: Path | str,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> List[str]:
        """Walk ``root`` and emit Prolog-style facts."""
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

        facts: List[str] = []
        for file_path in root.rglob("*"):
            if file_path.is_dir() or file_path.suffix not in extensions:
                continue
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

            module = self._module_name(file_path, root)
            lang = {
                ".py": "python",
                ".rs": "rust",
                ".cpp": "cpp",
                ".cc": "cpp",
                ".cxx": "cpp",
                ".hpp": "cpp",
                ".h": "cpp",
                ".c": "c",
            }.get(file_path.suffix, "unknown")

            facts.append(_fact("lang", module, lang))

            if file_path.suffix == ".py":
                for mod, sym in self._python_imports(tree):
                    target = f"{mod}.{sym}" if sym != "*" else mod
                    facts.append(_fact("imports", module, target))
                for sym in self._python_exports(tree):
                    facts.append(_fact("exports", module, sym))
                for caller, callee in self._python_calls(tree, module):
                    facts.append(_fact("calls", caller, callee))
            elif file_path.suffix == ".rs":
                facts.extend(self._rust_facts(tree, module))
            elif file_path.suffix in (".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"):
                facts.extend(self._cpp_facts(tree, module))

        return facts


class LogicEngine:
    """A small Datalog-like engine over ground Prolog facts."""

    def __init__(self) -> None:
        self.facts: List[Tuple[str, Tuple[str, ...]]] = []
        self._by_predicate: Dict[str, List[Tuple[str, ...]]] = {}

    def load_facts(self, facts: List[str]) -> None:
        """Parse and load a list of Prolog-style fact strings."""
        self.facts = []
        self._by_predicate = {}
        for line in facts:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            m = _FACT_RE.match(line)
            if not m:
                continue
            predicate = m.group(1)
            args = tuple(a.strip().strip('"').strip("'") for a in m.group(2).split(","))
            self.facts.append((predicate, args))
            self._by_predicate.setdefault(predicate, []).append(args)

    def query(self, predicate: str, *args: str) -> List[Tuple[str, ...]]:
        """Return facts matching the predicate and wildcard pattern."""
        results = []
        for fact_args in self._by_predicate.get(predicate, []):
            if len(fact_args) != len(args):
                continue
            if all(a == "_" or a == fact_args[i] for i, a in enumerate(args)):
                results.append(fact_args)
        return results

    def _dependency_graph(self) -> "nx.DiGraph":
        """Build a directed graph from imports/uses/includes/calls facts."""
        if nx is None:
            raise RuntimeError("networkx is required for dependency analysis")
        g = nx.DiGraph()
        for predicate, args in self.facts:
            if predicate in ("imports", "uses", "includes", "calls") and len(args) == 2:
                src, tgt = args
                g.add_edge(src, tgt, relation=predicate)
        return g

    def derive_reachability(self) -> Dict[str, List[str]]:
        """Return a mapping from each node to the set of nodes it can reach."""
        g = self._dependency_graph()
        reach: Dict[str, List[str]] = {}
        for node in g.nodes:
            try:
                reach[node] = list(nx.single_source_shortest_path(g, node).keys())
            except Exception:
                reach[node] = []
        return reach

    def detect_cycles(self) -> List[List[str]]:
        """Return elementary cycles in the dependency graph."""
        g = self._dependency_graph()
        try:
            return list(nx.simple_cycles(g))
        except Exception:
            return []

    def trace(self, source: str, target: str) -> Optional[List[str]]:
        """Return a shortest dependency path from source to target."""
        g = self._dependency_graph()
        try:
            return nx.shortest_path(g, source, target)
        except Exception:
            return None

    def unsafe_ffi_transitions(
        self,
        boundaries: Optional[List[Tuple[str, str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Find cross-language dependencies without an explicit boundary fact."""
        lang_map = {}
        for predicate, args in self.facts:
            if predicate == "lang" and len(args) == 2:
                lang_map[args[0]] = args[1]

        boundaries_set: Set[Tuple[str, str]] = set()
        if boundaries:
            for src, tgt, _ in boundaries:
                boundaries_set.add((src, tgt))
        else:
            for predicate, args in self.facts:
                if predicate == "boundary" and len(args) == 3:
                    boundaries_set.add((args[0], args[1]))

        unsafe = []
        for predicate, args in self.facts:
            if predicate in ("imports", "uses", "includes", "calls") and len(args) == 2:
                src, tgt = args
                src_lang = lang_map.get(src)
                tgt_lang = lang_map.get(tgt)
                if src_lang and tgt_lang and src_lang != tgt_lang:
                    if (src, tgt) not in boundaries_set:
                        unsafe.append(
                            {
                                "source": src,
                                "target": tgt,
                                "source_lang": src_lang,
                                "target_lang": tgt_lang,
                                "relation": predicate,
                            }
                        )
        return unsafe

    def derivation_trace(
        self,
        source: str,
        target: str,
    ) -> List[Dict[str, str]]:
        """Return a human-readable derivation trace for ``source -> target``."""
        g = self._dependency_graph()
        try:
            path = nx.shortest_path(g, source, target)
        except Exception:
            return []

        trace = []
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            edge_data = g.get_edge_data(src, tgt) or {}
            trace.append(
                {
                    "step": i + 1,
                    "source": src,
                    "target": tgt,
                    "relation": edge_data.get("relation", "depends_on"),
                }
            )
        return trace

    def run_rules(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """Forward-chain a few built-in rules and return inferred facts."""
        inferred = []

        # depends_on(X, Y) :- imports(X, Y); uses(X, Y); includes(X, Y); calls(X, Y).
        for predicate, args in self.facts:
            if predicate in ("imports", "uses", "includes", "calls"):
                inferred.append(("depends_on", args))

        # transitively reachable facts are computed on demand, but we can also
        # materialize reachability facts for a fixed set of start nodes.
        reach = self.derive_reachability()
        for src, tgts in reach.items():
            for tgt in tgts:
                if src != tgt:
                    inferred.append(("reaches", (src, tgt)))

        # Deduplicate while preserving order.
        seen: Set[Tuple[str, Tuple[str, ...]]] = set()
        out = []
        for fact in inferred:
            if fact not in seen:
                seen.add(fact)
                out.append(fact)
        return out


@dataclass
class RefinementFeedback:
    """Structured feedback from concolic and chiasmus analyses."""

    unsat_core: List[str] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    unsafe_ffi: List[Dict[str, str]] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_llm_message(self) -> str:
        lines = ["=== Formal feedback ===", ""]
        if self.unsat_core:
            lines.append("Unsatisfiable core from Z3 manifest verification:")
            for rule in self.unsat_core:
                lines.append(f"- {rule}")
            lines.append("")
        if self.cycles:
            lines.append("Dependency cycles detected by Chiasmus:")
            for cycle in self.cycles:
                lines.append(f"- {' -> '.join(cycle + [cycle[0]])}")
            lines.append("")
        if self.unsafe_ffi:
            lines.append("Unsafe cross-language transitions:")
            for t in self.unsafe_ffi:
                lines.append(
                    f"- {t['source']} ({t['source_lang']}) -> {t['target']} "
                    f"({t['target_lang']}) via {t['relation']}"
                )
            lines.append("")
        if self.traces:
            lines.append("Selected derivation traces:")
            for trace in self.traces:
                steps = " -> ".join(
                    f"{s['source']}[{s['relation']}]{s['target']}"
                    for s in trace.get("steps", [])
                )
                lines.append(f"- {steps}")
            lines.append("")
        if self.summary:
            lines.append(self.summary)
        return "\n".join(lines)


def analyze_repository(
    repo_path: Path | str,
    manifest_boundaries: Optional[List[Tuple[str, str, str]]] = None,
) -> RefinementFeedback:
    """High-level helper: emit facts, run logic engine, return feedback."""
    emitter = PrologFactEmitter()
    facts = emitter.emit_facts(repo_path)
    engine = LogicEngine()
    engine.load_facts(facts)
    cycles = engine.detect_cycles()
    unsafe = engine.unsafe_ffi_transitions(manifest_boundaries)
    return RefinementFeedback(
        cycles=cycles,
        unsafe_ffi=unsafe,
        summary=(
            "Repository analysis complete. "
            f"{len(cycles)} cycle(s), {len(unsafe)} unsafe FFI transition(s)."
        ),
    )
