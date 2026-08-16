"""Compatibility re-export of the canonical concolic manifest verifier.

The canonical implementation lives in ``aero_forge.concolic`` and is imported here
so existing code and tests that use ``aero_forge.builder.concolic`` continue to work.
"""

from __future__ import annotations

from aero_forge.concolic import (
    ConcolicManifestVerifier,
    ConcolicResult,
    verify_manifest,
)

__all__ = ["ConcolicManifestVerifier", "ConcolicResult", "verify_manifest"]
