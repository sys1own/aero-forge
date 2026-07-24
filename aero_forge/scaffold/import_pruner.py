"""Static import pruning for generated Python modules."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_DYNAMIC_SENTINELS = frozenset({"__import__", "eval", "exec", "globals", "vars"})


@dataclass
class PruneOutcome:
    """Result of a single-module import-pruning pass."""

    kept_imports: List[ast.stmt] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    skipped_dynamic: bool = False
    source: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.pruned)


def _bound_name(alias: ast.alias) -> str:
    if alias.asname:
        return alias.asname
    return alias.name.split(".")[0]


def _has_dynamic_lookup(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                return True
            if node.attr == "import_module":
                return True
        elif isinstance(node, ast.Name) and node.id in _DYNAMIC_SENTINELS:
            return True
    return False


def _used_names(tree: ast.AST) -> Set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _string_tokens(tree: ast.AST) -> Set[str]:
    tokens: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.update(_IDENTIFIER_RE.findall(node.value))
    return tokens


def prune_dead_imports(module_ast: ast.Module) -> PruneOutcome:
    """Strip imports whose bound names are unused in ``module_ast``."""
    body: Sequence[ast.stmt] = getattr(module_ast, "body", []) or []
    top_imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not top_imports:
        return PruneOutcome(kept_imports=[], pruned=[], skipped_dynamic=False)

    if _has_dynamic_lookup(module_ast):
        return PruneOutcome(
            kept_imports=list(top_imports), pruned=[], skipped_dynamic=True
        )

    safe_used = _used_names(module_ast) | _string_tokens(module_ast)

    kept: List[ast.stmt] = []
    pruned: List[str] = []

    for node in top_imports:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            kept.append(node)
            continue
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            kept.append(node)
            continue

        keep_aliases = [a for a in node.names if _bound_name(a) in safe_used]
        dead_aliases = [a for a in node.names if _bound_name(a) not in safe_used]
        pruned.extend(_bound_name(a) for a in dead_aliases)

        if not keep_aliases:
            continue
        if len(keep_aliases) == len(node.names):
            kept.append(node)
            continue

        if isinstance(node, ast.Import):
            rebuilt: ast.stmt = ast.Import(names=keep_aliases)
        else:
            rebuilt = ast.ImportFrom(
                module=node.module, names=keep_aliases, level=node.level
            )
        ast.copy_location(rebuilt, node)
        kept.append(rebuilt)

    return PruneOutcome(kept_imports=kept, pruned=pruned, skipped_dynamic=False)


def prune_source(source: str) -> PruneOutcome:
    """Parse *source*, prune dead imports, and return a source-aware outcome."""
    tree = ast.parse(source)
    outcome = prune_dead_imports(tree)
    if outcome.changed and hasattr(ast, "unparse"):
        body = tree.body
        kept_set = set(id(k) for k in outcome.kept_imports)
        new_body = [
            (node if id(node) in kept_set or not isinstance(node, (ast.Import, ast.ImportFrom)) else None)
            for node in body
        ]
        tree.body = [n for n in new_body if n is not None]
        outcome.source = ast.unparse(tree)
    else:
        outcome.source = source
    return outcome


def render_imports(nodes: Sequence[ast.stmt]) -> List[str]:
    if hasattr(ast, "unparse"):
        return [ast.unparse(node) for node in nodes]
    return [ast.dump(node) for node in nodes]
