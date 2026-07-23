import numpy as np
from generated import vector_dot


def test_vector_dot():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert vector_dot(a, b) == 32.0
