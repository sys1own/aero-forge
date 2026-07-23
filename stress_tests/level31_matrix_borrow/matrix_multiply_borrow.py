def matrix_multiply_borrow(a, b):
    result = []
    for i in range(len(a)):
        ai = a[i]
        row = []
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(b)):
                s += ai[k] * b[k][j]
            row.append(s)
        result.append(row)
    return result
