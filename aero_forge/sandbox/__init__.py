"""Sandbox utilities for isolated test execution and web session isolation."""

from aero_forge.sandbox.manager import (
    ExecutionTrace,
    Sandbox,
    SandboxManager,
    TraceVerifier,
)
from aero_forge.errors import SemanticRegressionError

__all__ = [
    "ExecutionTrace",
    "Sandbox",
    "SandboxManager",
    "SemanticRegressionError",
    "TraceVerifier",
]
