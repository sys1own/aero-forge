def dot_product_zip(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s
