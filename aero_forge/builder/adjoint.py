"""Compatibility re-export of the canonical category-theoretic bootstrapper.

The canonical implementation lives in ``aero_forge.adjoint`` and is imported here
so existing code and tests that use ``aero_forge.builder.adjoint`` continue to work.
"""

from __future__ import annotations

from aero_forge.adjoint import NodeStub, SchemaBootstrapper

__all__ = ["NodeStub", "SchemaBootstrapper"]
