def dot_product_index(a, b):
    s = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        s += a[i] * b[i]
    return s
