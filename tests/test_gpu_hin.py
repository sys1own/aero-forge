"""Tests for GPU kernel lowering from HIN numeric kernels."""

import pytest

from aero_forge.errors import UnsupportedError
from aero_forge.gpu import find_gpu_functions, lower_hin_to_cuda, schedule_gpu_grid


def test_find_gpu_functions():
    src = "# @accelerate gpu\ndef f(x):\n    return x[i] + 1.0\n"
    assert find_gpu_functions(src) == ["f"]


def test_schedule_gpu_grid():
    assert schedule_gpu_grid(100, 256) == (1, 256)
    assert schedule_gpu_grid(1024, 256) == (4, 256)
    assert schedule_gpu_grid(0, 256) == (0, 256)


def test_lower_hin_to_cuda():
    src = "\n# @accelerate gpu\ndef scale(x):\n    return x[i] * 2.0 + 1.0\n"
    cu = lower_hin_to_cuda(src, "scale")
    assert "__global__ void scale" in cu
    assert "y[i] = ((x[i] * 2.0) + 1.0)" in cu
    assert "blockDim.x" in cu


def test_lower_hin_to_cuda_unsupported_body():
    with pytest.raises(UnsupportedError):
        lower_hin_to_cuda("def f(x):\n    return x\n", "f")
