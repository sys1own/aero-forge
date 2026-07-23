def sum_pairs(a, b):
    total = 0
    for x, y in zip(a, b):
        total += x + y
    return total
