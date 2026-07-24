"""Library of reference algorithm implementations for prompt-driven generation."""

from __future__ import annotations

import importlib.resources as pkg_resources
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Algorithm:
    """A reference algorithm with metadata."""

    name: str
    category: str
    path: Path
    source: str
    metadata: Dict[str, object]


_ALGORITHMS: Optional[Dict[str, Algorithm]] = None


def _scan_algorithms(root_name: str = "aero_forge.algorithms") -> Dict[str, Algorithm]:
    """Recursively discover all algorithm modules under ``root_name``."""
    algorithms: Dict[str, Algorithm] = {}
    try:
        root_files = pkg_resources.files(root_name)
    except (ImportError, TypeError):
        return algorithms

    def recurse(pkg_path: str, files):
        for item in files.iterdir():
            if item.is_dir():
                subpkg = f"{pkg_path}.{item.name}"
                try:
                    recurse(subpkg, pkg_resources.files(subpkg))
                except (ImportError, TypeError):
                    pass
            elif (
                item.is_file()
                and item.name.endswith(".py")
                and item.name != "__init__.py"
            ):
                name = item.stem
                path = Path(str(item))
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                metadata = _extract_metadata(text)
                if metadata.get("name", name) in algorithms:
                    continue
                algorithms[metadata.get("name", name)] = Algorithm(
                    name=metadata.get("name", name),
                    category=metadata.get("category", "general"),
                    path=path,
                    source=text,
                    metadata=metadata,
                )

    recurse(root_name, root_files)
    return algorithms


def _load_algorithms() -> Dict[str, Algorithm]:
    """Load and cache all algorithms."""
    global _ALGORITHMS
    if _ALGORITHMS is not None:
        return _ALGORITHMS
    _ALGORITHMS = _scan_algorithms()
    return _ALGORITHMS


def _extract_metadata(source: str) -> Dict[str, object]:
    """Return the ``METADATA`` dict from an algorithm module, if present."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "METADATA":
                    try:
                        return ast.literal_eval(node.value)  # type: ignore[arg-type]
                    except Exception:
                        return {}
    return {}


def list_algorithms(category: Optional[str] = None) -> List[str]:
    """Return the names of available reference algorithms."""
    algorithms = _load_algorithms()
    if category:
        return sorted(a.name for a in algorithms.values() if a.category == category)
    return sorted(algorithms.keys())


def list_categories() -> List[str]:
    """Return the distinct algorithm categories."""
    algorithms = _load_algorithms()
    return sorted({a.category for a in algorithms.values()})


def get_algorithm(name: str) -> Optional[Algorithm]:
    """Return an ``Algorithm`` by name, or None."""
    return _load_algorithms().get(name)


def get_algorithm_source(name: str) -> Optional[str]:
    """Return the source text for a reference algorithm, or None."""
    algo = get_algorithm(name)
    return algo.source if algo else None


def find_algorithm(prompt: str) -> Optional[Algorithm]:
    """Return the best-matching reference algorithm for a prompt using keywords."""
    lowered = prompt.lower()
    keywords = {
        "fibonacci": ["fibonacci"],
        "gcd": ["gcd", "greatest common divisor"],
        "is_prime": ["prime", "primality"],
        "matrix_multiply": ["matrix", "matrices", "multiply"],
        "naive_multiply": ["matrix", "multiply"],
        "blocked_multiply": ["matrix", "cache", "block"],
        "strassen": ["matrix", "fast matrix", "strassen"],
        "quicksort": ["sort", "quicksort", "quick sort"],
        "mergesort": ["sort", "mergesort", "merge sort", "stable"],
        "insertion_sort": ["sort", "insertion", "small"],
        "selection_sort": ["sort", "selection"],
        "heap_sort": ["sort", "heap"],
        "timsort": ["sort", "timsort", "real-world"],
        "cooley_tukey": ["fft", "dft", "fourier", "signal"],
        "binary_search": ["search", "binary", "lookup"],
        "mandelbrot": ["mandelbrot"],
    }
    for name, terms in keywords.items():
        if any(term in lowered for term in terms):
            return get_algorithm(name)
    return None


def select_algorithm(
    prompt: str,
    category: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    config_override: Optional["ConfigOverride"] = None,
) -> Optional[Algorithm]:
    """Select the best algorithm for a prompt.

    Uses an LLM when a provider is configured, otherwise falls back to keyword
    matching via ``find_algorithm``.
    """
    candidates = _load_algorithms()
    if category:
        candidates = {k: v for k, v in candidates.items() if v.category == category}
    if not candidates:
        return None

    if llm_provider:
        from aero_forge.llm.clients import get_llm_client

        client = get_llm_client(
            llm_provider, model=model, config_override=config_override
        )
        if client is not None:
            descriptions = []
            for algo in candidates.values():
                meta = algo.metadata
                complexity = meta.get("complexity", {})
                desc = (
                    f"- {algo.name}: {meta.get('use_cases', [])}; "
                    f"time {complexity.get('time', 'unknown')}, "
                    f"space {complexity.get('space', 'unknown')}; "
                    f"constraints: {meta.get('constraints', [])}"
                )
                descriptions.append(desc)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an algorithm expert. Select exactly one "
                        "algorithm name from the list that best fits the request. "
                        "Return only the algorithm name."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Request: {prompt}\n\n"
                        f"Available algorithms:\n{chr(10).join(descriptions)}\n\n"
                        "Selected algorithm name:"
                    ),
                },
            ]
            response = client.generate(messages, temperature=0.2)
            if response:
                name = response.strip().splitlines()[0].strip().strip("`'")
                if name in candidates:
                    return candidates[name]

    return find_algorithm(prompt)


def algorithm_prompt_context(
    prompt: str,
    selected: Optional[Algorithm] = None,
    algorithm_library: bool = False,
) -> str:
    """Return a string with reference algorithm source for the LLM context."""
    if selected is None and algorithm_library:
        selected = find_algorithm(prompt)
    if not selected:
        return ""
    return (
        f"Selected reference algorithm: {selected.name}\n"
        f"Category: {selected.category}\n"
        f"Complexity: {selected.metadata.get('complexity', {})}\n"
        f"Use cases: {selected.metadata.get('use_cases', [])}\n"
        f"Constraints: {selected.metadata.get('constraints', [])}\n\n"
        f"Use the implementation below as a starting point and adapt it to the "
        f"request (do not copy it verbatim if it does not exactly match):\n\n"
        f"```python\n{selected.source}\n```\n"
    )
