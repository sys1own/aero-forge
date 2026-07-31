"""Python bridge to the Rust HIN (MELL interaction net) engine."""

import json
import os
from typing import Any, Dict, Optional

try:
    from aero_forge_native import HinEngine  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - native module is optional at import time
    try:
        from aero_forge._native import HinEngine  # type: ignore[attr-defined]
    except Exception:
        HinEngine = None  # type: ignore[misc,assignment]


# MELL structural type helpers used by the emitters.
class MELLType:
    """MELL-typed wire annotation."""

    def __init__(self, kind: str, left=None, right=None, inner=None):
        self.kind = kind
        self.left = left
        self.right = right
        self.inner = inner

    @staticmethod
    def unit():
        return MELLType("I")

    @staticmethod
    def any_():
        return MELLType("Any")

    @staticmethod
    def bang(inner):
        return MELLType("Bang", inner=inner)

    @staticmethod
    def implication(left, right):
        return MELLType("Implication", left=left, right=right)

    @staticmethod
    def tensor(left, right):
        return MELLType("Tensor", left=left, right=right)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        if self.left is not None:
            d["left"] = self.left.to_dict() if isinstance(self.left, MELLType) else self.left
        if self.right is not None:
            d["right"] = (
                self.right.to_dict() if isinstance(self.right, MELLType) else self.right
            )
        if self.inner is not None:
            d["inner"] = (
                self.inner.to_dict() if isinstance(self.inner, MELLType) else self.inner
            )
        return d


def reduce_uast(uast: Any, max_steps: int = 1_000_000) -> Dict[str, Any]:
    """Build and reduce a HIN graph from a UAST value.

    Returns ``{"steps": int, "graph": list}``.  Falls back to a no-op
    dictionary when the native module is unavailable so the bridge can be
    imported everywhere.
    """
    if HinEngine is None:
        return {"steps": 0, "graph": [], "native": False}

    engine = HinEngine()
    engine.build_from_json(json.dumps(uast))
    steps = engine.reduce_to_completion(max_steps)
    return {"steps": steps, "graph": json.loads(engine.to_json()), "native": True}


def native_available() -> bool:
    """Return ``True`` when the Rust HIN engine extension is importable."""
    return HinEngine is not None


# Feature-flag guard for integration with the translator/emitters.
HIN_ENGINE_ENABLED = os.environ.get("AERO_HIN_ENGINE", "1") == "1"
