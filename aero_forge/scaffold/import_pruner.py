"""Static import pruning and typing-import injection for generated Python modules."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_DYNAMIC_SENTINELS = frozenset({"__import__", "eval", "exec", "globals", "vars"})

TYPING_NAMES = frozenset({
    "Any",
    "Callable",
    "Dict",
    "Iterable",
    "List",
    "Optional",
    "Sequence",
    "Tuple",
    "TypeVar",
    "Union",
})


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


def _annotation_nodes(tree: ast.Module) -> List[ast.AST]:
    """Yield every AST node that is a Python type annotation in *tree*."""
    nodes: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.annotation:
            nodes.append(node.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                nodes.append(node.returns)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    nodes.append(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                nodes.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                nodes.append(node.args.kwarg.annotation)
        if isinstance(node, ast.arg) and node.annotation:
            nodes.append(node.annotation)
    return nodes


def _collect_typing_names(node: ast.AST) -> Set[str]:
    """Recursively collect ``typing`` names referenced in an annotation node."""
    found: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in TYPING_NAMES:
            found.add(child.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # ``from __future__ import annotations`` stores annotations as strings.
            try:
                expr = ast.parse(child.value, mode="eval")
            except SyntaxError:
                continue
            found |= _collect_typing_names(expr)
    return found


def _typing_imports_in_source(source: str) -> Set[str]:
    """Return the typing names already available in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imported: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name == "*":
                    return set(TYPING_NAMES)
                imported.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "typing" or alias.asname == "typing":
                    return set(TYPING_NAMES)
    return imported


def _insert_import_line(source: str, import_line: str) -> str:
    """Insert *import_line* after the shebang / module docstring / leading comments."""
    lines = source.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    while insert_at < len(lines) and lines[insert_at].strip().startswith("#"):
        insert_at += 1
    if insert_at < len(lines) and lines[insert_at].startswith('"""'):
        single_line_doc = lines[insert_at].strip().endswith('"""') and lines[insert_at].count('"""') == 2
        if not single_line_doc:
            insert_at += 1
            while insert_at < len(lines) and '"""' not in lines[insert_at]:
                insert_at += 1
            insert_at += 1
        else:
            insert_at += 1
    if insert_at > 0 and insert_at < len(lines) and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, import_line + "\n")
    return "".join(lines)


def ensure_typing_imports(source: str, extra_names: Optional[Set[str]] = None) -> str:
    """Inject ``from typing import ...`` for any typing symbols used but not imported.

    Scans function annotations, variable type hints, argument types, and type
    aliases (including stringified annotations under ``from __future__ import
    annotations``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    used: Set[str] = set()
    for ann in _annotation_nodes(tree):
        used |= _collect_typing_names(ann)
    if extra_names:
        used |= extra_names
    if not used:
        return source

    imported = _typing_imports_in_source(source)
    missing = used - imported
    if not missing:
        return source

    import_line = "from typing import " + ", ".join(sorted(missing))
    return _insert_import_line(source, import_line)


def render_imports(nodes: Sequence[ast.stmt]) -> List[str]:
    if hasattr(ast, "unparse"):
        return [ast.unparse(node) for node in nodes]
    return [ast.dump(node) for node in nodes]
