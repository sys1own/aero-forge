from typing import List
import cmath

METADATA = {
    "name": "cooley_tukey",
    "category": "fft",
    "complexity": {"time": "O(n log n)", "space": "O(n)"},
    "use_cases": ["signal processing", "polynomial multiplication"],
    "constraints": ["input length must be a power of two"],
}


def cooley_tukey(x: List[complex]) -> List[complex]:
    n = len(x)
    if n == 1:
        return x
    even = cooley_tukey(x[0::2])
    odd = cooley_tukey(x[1::2])
    result = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    return result
