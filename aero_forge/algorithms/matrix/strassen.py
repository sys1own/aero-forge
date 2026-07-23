from typing import List

METADATA = {
    "name": "strassen",
    "category": "matrix",
    "complexity": {"time": "O(n^log2(7)) ~ O(n^2.81)", "space": "O(n^2)"},
    "use_cases": ["large dense matrices", "theoretical speedup"],
    "constraints": ["overhead for small matrices", "only square power-of-two"],
}


def strassen(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    if n <= 64:
        return _naive(a, b)
    mid = n // 2
    a11 = [row[:mid] for row in a[:mid]]
    a12 = [row[mid:] for row in a[:mid]]
    a21 = [row[:mid] for row in a[mid:]]
    a22 = [row[mid:] for row in a[mid:]]
    b11 = [row[:mid] for row in b[:mid]]
    b12 = [row[mid:] for row in b[:mid]]
    b21 = [row[:mid] for row in b[mid:]]
    b22 = [row[mid:] for row in b[mid:]]
    m1 = strassen(_add(a11, a22), _add(b11, b22))
    m2 = strassen(_add(a21, a22), b11)
    m3 = strassen(a11, _sub(b12, b22))
    m4 = strassen(a22, _sub(b21, b11))
    m5 = strassen(_add(a11, a12), b22)
    m6 = strassen(_sub(a21, a11), _add(b11, b12))
    m7 = strassen(_sub(a12, a22), _add(b21, b22))
    c11 = _add(_sub(_add(m1, m4), m5), m7)
    c12 = _add(m3, m5)
    c21 = _add(m2, m4)
    c22 = _add(_sub(m1, m2), _add(m3, m6))
    return _combine(c11, c12, c21, c22)


def _add(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    return [[x[i][j] + y[i][j] for j in range(len(x))] for i in range(len(x))]


def _sub(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    return [[x[i][j] - y[i][j] for j in range(len(x))] for i in range(len(x))]


def _naive(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _combine(c11, c12, c21, c22):
    n = len(c11)
    result = [[0.0 for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = c11[i][j]
            result[i][j + n] = c12[i][j]
            result[i + n][j] = c21[i][j]
            result[i + n][j + n] = c22[i][j]
    return result
