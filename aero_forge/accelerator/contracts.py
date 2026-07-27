"""Abstract contracts for the dual-engine accelerator surface."""

from abc import ABC, abstractmethod
from typing import Dict, List


class HasherABC(ABC):
    """BLAKE3-style incremental hasher contract."""

    @abstractmethod
    def update(self, data: bytes) -> None:
        """Feed *data* into the hasher."""

    @abstractmethod
    def finalize(self) -> str:
        """Return the lower-case hexadecimal digest of all bytes hashed so far."""

    @abstractmethod
    def digest(self) -> bytes:
        """Return the raw digest bytes."""

    @abstractmethod
    def copy(self) -> "HasherABC":
        """Return a copy of the hasher with identical internal state."""


class GraphEngineABC(ABC):
    """Dependency-graph primitives contract backed by an immutable graph."""

    @abstractmethod
    def topological_sort(self) -> List[str]:
        """Return a valid topological ordering of the graph provided at construction."""

    @abstractmethod
    def prune_unreachable(self, roots: List[str]) -> List[str]:
        """Return the nodes reachable from *roots*, sorted deterministically."""
