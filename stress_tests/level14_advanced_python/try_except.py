def try_func(x: int) -> int:
    try:
        return x // 0
    except ZeroDivisionError:
        return 0
