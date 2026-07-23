def matrix_multiply(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    result = []
    for i in range(n):
        row = []
        for j in range(m):
            total = 0
            for k in range(p):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)
    return result
