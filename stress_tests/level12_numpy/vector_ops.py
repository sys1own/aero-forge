import numpy as np


def scale_array(arr: np.ndarray) -> np.ndarray:
    return arr * 2.0 + 1.0


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)


def sum_array(arr: np.ndarray) -> float:
    return np.sum(arr)
