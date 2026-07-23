def matrix_multiply_nested(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            row.append(s)
        result.append(row)
    return result
