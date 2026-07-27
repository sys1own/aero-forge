"""Pure-Python BLAKE3 hasher using the reference ``blake3`` package."""

import os
from typing import Union

from aero_forge.accelerator.contracts import HasherABC

try:
    import blake3 as _blake3
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The 'blake3' package is required for the fallback hasher. "
        "Install it with 'pip install blake3'."
    ) from exc


class Hasher(HasherABC):
    """BLAKE3 hasher with an API that mirrors the Rust/PyO3 implementation."""

    __slots__ = ("_hasher",)

    def __init__(self) -> None:
        self._hasher = _blake3.blake3()

    def update(self, data: Union[bytes, bytearray, memoryview]) -> None:
        self._hasher.update(data)

    def finalize(self) -> str:
        return self._hasher.hexdigest()

    def digest(self) -> bytes:
        return self._hasher.digest()

    def copy(self) -> "Hasher":
        clone = Hasher()
        clone._hasher = self._hasher.copy()
        return clone


def hash_bytes(data: Union[bytes, bytearray, memoryview]) -> str:
    return _blake3.blake3(data).hexdigest()


def hash_file(path: Union[str, os.PathLike[str]]) -> str:
    hasher = _blake3.blake3()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
