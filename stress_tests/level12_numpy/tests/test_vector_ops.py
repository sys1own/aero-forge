import numpy as np
from vector_ops import scale_array, dot_product, sum_array


def test_scale_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = scale_array(arr)
    assert result == [3.0, 5.0, 7.0]


def test_dot_product():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert dot_product(a, b) == 32.0


def test_sum_array():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    assert sum_array(arr) == 10.0
