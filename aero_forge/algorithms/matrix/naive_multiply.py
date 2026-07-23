from typing import List

METADATA = {
    "name": "naive_multiply",
    "category": "matrix",
    "complexity": {"time": "O(n^3)", "space": "O(n^2)"},
    "use_cases": ["small dense matrices", "reference implementation"],
    "constraints": ["cubic time"],
}


def naive_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result
