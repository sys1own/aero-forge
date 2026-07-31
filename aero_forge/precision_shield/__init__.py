"""Aero-Forge precision shield: Rust type and trait selection."""

from .precision_rs import ensure_precision_traits
from .rust_shield import RustSemanticShield
from .shield import Shield

__all__ = ["ensure_precision_traits", "RustSemanticShield", "Shield"]
