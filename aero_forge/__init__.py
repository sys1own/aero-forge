"""Aero-Forge: LLM-integrated Python to Rust build tool."""

__version__ = "0.1.0"

from . import (
    config,
    errors,
    hin_vm,
    precision_shield,
    scaffold,
    translator,
)
from .native_bridge import NativeAccelerator, accelerate
from .orchestrator.orchestrator import Orchestrator
from .sandbox.manager import Sandbox

__all__ = [
    "accelerate",
    "NativeAccelerator",
    "Orchestrator",
    "Sandbox",
    "config",
    "errors",
    "hin_vm",
    "precision_shield",
    "scaffold",
    "translator",
]
