def matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                result[i][j] += a[i][k] * b[k][j]
    return result
