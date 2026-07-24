from campaign import (
    cooley_tukey,
    dot_product,
    fibonacci,
    mandelbrot,
    matrix_multiply,
    most_frequent,
    primes_up_to,
    timsort,
)


def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765


def test_primes_up_to():
    assert primes_up_to(0) == []
    assert primes_up_to(1) == []
    assert primes_up_to(2) == [2]
    assert primes_up_to(10) == [2, 3, 5, 7]
    assert primes_up_to(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert primes_up_to(-5) == []


def test_matrix_multiply():
    assert matrix_multiply([], [[1.0]]) == []
    assert matrix_multiply([[1.0, 0.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]) == [
        [2.0, 3.0],
        [4.0, 5.0],
    ]
    assert matrix_multiply([[1.0, 2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]) == [
        [40.0, 46.0]
    ]
    assert matrix_multiply([[3.0]], [[4.0]]) == [[12.0]]


def test_mandelbrot():
    assert mandelbrot(0.0, 0.0, 100) == 100
    assert mandelbrot(-0.9, 0.0, 100) == 100
    assert mandelbrot(2.0, 0.0, 100) == 0
    assert mandelbrot(1.0, 0.0, 100) == 3


def test_timsort():
    assert timsort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
    assert timsort([]) == []
    assert timsort([5]) == [5]


def test_dot_product():
    assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    assert dot_product([], []) == 0.0


def test_cooley_tukey():
    real = [1.0, 0.0]
    imag = [0.0, 0.0]
    r, i = cooley_tukey(real, imag)
    assert len(r) == 2
    assert len(i) == 2


def test_most_frequent():
    assert most_frequent([1, 3, 1, 3, 2, 1]) == 1
    assert most_frequent([2, 2, 3, 3, 3, 1]) == 3
    assert most_frequent([]) == -1
