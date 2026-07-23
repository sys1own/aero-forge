"""Experimental GPU backend scaffolding.

Aero-Forge can detect functions annotated with ``# @accelerate gpu`` and route
them through a GPU backend.  This module provides the dispatcher and a minimal
CPU fallback for environments without a CUDA toolkit.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional

from aero_forge.errors import UnsupportedError

logger = logging.getLogger("aero_forge.gpu")

ACCELERATE_PATTERN = re.compile(
    r"^\s*#\s*@accelerate\s+(gpu)(?:\s|$)", re.IGNORECASE | re.MULTILINE
)


def has_gpu_pragma(source_text: str) -> bool:
    """Return True if the source contains a ``# @accelerate gpu`` pragma."""
    return ACCELERATE_PATTERN.search(source_text) is not None


def find_gpu_functions(source_text: str) -> List[str]:
    """Return the names of functions that follow a ``# @accelerate gpu`` pragma."""
    names: List[str] = []
    lines = source_text.splitlines()
    for i, line in enumerate(lines):
        if ACCELERATE_PATTERN.search(line):
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
                if match:
                    names.append(match.group(1))
                break
    return names


def nvcc_path() -> Optional[str]:
    """Return the path to ``nvcc`` if it is available, otherwise None."""
    return shutil.which("nvcc")


def compile_gpu_kernel(source_path: Path, function_names: List[str]) -> Optional[Path]:
    """Attempt to compile a GPU kernel for the marked functions.

    If ``nvcc`` is not installed, returns ``None`` and the build falls back to
    the CPU backend.  If ``nvcc`` is installed but the function cannot yet be
    lowered to CUDA, an ``UnsupportedError`` is raised.
    """
    if not nvcc_path():
        logger.warning(
            "GPU acceleration requested for %s but nvcc was not found; "
            "falling back to CPU compilation.",
            function_names,
        )
        return None

    # TODO: implement CUDA C kernel generation for numeric Python functions.
    raise UnsupportedError(
        "GPU kernel generation is not yet implemented for these functions: "
        f"{', '.join(function_names)}"
    )
