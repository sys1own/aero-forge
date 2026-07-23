def power(base: float, exp: int) -> float:
    result = 1.0
    for _ in range(exp):
        result *= base
    return result
