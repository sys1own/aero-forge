"""Invisible configuration layer for aero-forge."""

from aero_forge.invisible_config.engine import InferredTarget, InvisibleConfigEngine
from aero_forge.invisible_config.lean_parser import (
    LeanBlueprint,
    LeanBlueprintError,
    looks_like_lean_blueprint,
    parse_lean_blueprint,
)

__all__ = [
    "InvisibleConfigEngine",
    "InferredTarget",
    "LeanBlueprint",
    "LeanBlueprintError",
    "looks_like_lean_blueprint",
    "parse_lean_blueprint",
]

__version__ = "1.0.0"
