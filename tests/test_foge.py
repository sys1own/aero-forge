"""Unit tests for Fock-Space Graph Encoder (FoGE)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aero_forge.builder.foge import FockGraphEncoder


def test_circular_convolve_is_commutative():
    enc = FockGraphEncoder(dim=64)
    a = enc.vector("a")
    b = enc.vector("b")
    ab = enc.circular_convolve(a, b)
    ba = enc.circular_convolve(b, a)
    assert np.allclose(ab, ba)


def test_token_is_reversible_by_correlation():
    enc = FockGraphEncoder(dim=128)
    src = enc.vector("src")
    rel = enc.vector("imports")
    dst = enc.vector("dst")
    token = enc.token_for("src", "imports", "dst")
    # src ⊗ rel ⊗ dst correlated with rel ⊗ dst should recover src.
    probe = enc.circular_convolve(rel, dst)
    recovered = enc.circular_correlate(token, probe)
    assert np.corrcoef(recovered, src)[0, 1] > 0.3


def test_encode_repository_extracts_dependencies(tmp_path: Path) -> None:
    (tmp_path / "aero_forge").mkdir()
    (tmp_path / "aero_forge" / "__init__.py").write_text("from .core import run\n")
    (tmp_path / "aero_forge" / "core.py").write_text("import numpy\n")
    (tmp_path / "main.py").write_text("from aero_forge import core\n")

    enc = FockGraphEncoder(dim=64)
    result = enc.encode_repository(tmp_path)

    assert result["dim"] == 64
    assert "main" in result["nodes"]
    assert "aero_forge.core" in result["nodes"]
    assert len(result["tokens"]) == len(result["edges"]) > 0

    relations = {e["relation"] for e in result["edges"]}
    assert "imports" in relations


def test_vector_is_deterministic():
    enc1 = FockGraphEncoder(dim=64, seed=0)
    enc2 = FockGraphEncoder(dim=64, seed=0)
    assert np.allclose(enc1.vector("x"), enc2.vector("x"))


def test_similarity_of_identical_vectors():
    enc = FockGraphEncoder(dim=64)
    v = enc.vector("same")
    assert enc.similarity(v, v) == pytest.approx(1.0)
