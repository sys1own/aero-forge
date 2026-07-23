"""Library of reference algorithm implementations for prompt-driven generation."""

from __future__ import annotations

import importlib.resources as pkg_resources
import re
from pathlib import Path
from typing import Dict, List, Optional

_ALGORITHMS: Optional[Dict[str, str]] = None


def _load_algorithms() -> Dict[str, str]:
    """Load all `.py` algorithm files shipped in this package."""
    global _ALGORITHMS
    if _ALGORITHMS is not None:
        return _ALGORITHMS

    algorithms: Dict[str, Path] = {}
    try:
        files = pkg_resources.files("aero_forge.algorithms")
    except (ImportError, TypeError):
        _ALGORITHMS = {}
        return _ALGORITHMS

    for item in files.iterdir():
        if item.is_file() and item.name.endswith(".py") and item.name != "__init__.py":
            name = item.stem
            algorithms[name] = Path(str(item))

    _ALGORITHMS = algorithms
    return _ALGORITHMS


def list_algorithms() -> List[str]:
    """Return the names of available reference algorithms."""
    return sorted(_load_algorithms().keys())


def get_algorithm(name: str) -> Optional[str]:
    """Return the source text for a reference algorithm, or None."""
    path = _load_algorithms().get(name)
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def find_algorithm(prompt: str) -> Optional[str]:
    """Return the source text of the best-matching reference algorithm for a prompt."""
    lowered = prompt.lower()
    keywords = {
        "fibonacci": ["fibonacci"],
        "gcd": ["gcd", "greatest common divisor"],
        "is_prime": ["prime"],
        "matrix_multiply": ["matrix", "matrices"],
        "quicksort": ["sort", "quicksort", "quick sort"],
        "mandelbrot": ["mandelbrot"],
    }
    for name, terms in keywords.items():
        if any(term in lowered for term in terms):
            return get_algorithm(name)
    return None


def algorithm_prompt_context(prompt: str) -> str:
    """Return a string with reference algorithm source, or an empty string."""
    source = find_algorithm(prompt)
    if not source:
        return ""
    return (
        "You may use the following reference implementation as inspiration "
        "(do not copy it verbatim unless it exactly matches the request):\n\n"
        f"```python\n{source}\n```\n"
    )
