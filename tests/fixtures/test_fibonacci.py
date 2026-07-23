from fibonacci import fibonacci


def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_values():
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
