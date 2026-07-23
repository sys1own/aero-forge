def sum_of_squares(n: int) -> int:
    total = 0
    for i in range(n + 1):
        total += i * i
    return total
