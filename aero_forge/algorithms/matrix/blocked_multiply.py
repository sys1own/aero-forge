from typing import List

METADATA = {
    "name": "blocked_multiply",
    "category": "matrix",
    "complexity": {"time": "O(n^3)", "space": "O(n^2)"},
    "use_cases": ["cache-friendly matrix multiplication"],
    "constraints": ["block size tuning required"],
}


def blocked_multiply(
    a: List[List[float]], b: List[List[float]], block: int = 32
) -> List[List[float]]:
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for ii in range(0, rows_a, block):
        for jj in range(0, cols_b, block):
            for kk in range(0, cols_a, block):
                for i in range(ii, min(ii + block, rows_a)):
                    for j in range(jj, min(jj + block, cols_b)):
                        s = 0.0
                        for k in range(kk, min(kk + block, cols_a)):
                            s += a[i][k] * b[k][j]
                        result[i][j] += s
    return result
