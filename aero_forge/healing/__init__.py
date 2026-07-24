"""Aero-Forge deterministic healing layer.

Healing is performed by static AST rewrites and pattern matching. No LLM
calls are made from this package; LLM interaction is confined to upstream
intent parsing and human-facing diagnostics.
"""

from aero_forge.healing.router import try_auto_fix

__all__ = ["try_auto_fix"]
