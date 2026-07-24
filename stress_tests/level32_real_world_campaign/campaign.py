from typing import List
import math


def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number efficiently using iteration."""
    if n <= 1:
        return n
    a = 0
    b = 1
    for _ in range(2, n + 1):
        temp = a + b
        a = b
        b = temp
    return b


def primes_up_to(n: int) -> list[int]:
    """Return a list of all prime numbers <= n using the Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = False
                j = j + i
        i = i + 1
    result = []
    k = 2
    while k <= n:
        if is_prime[k]:
            result.append(k)
        k = k + 1
    return result


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices a and b. Returns empty list if either input is empty."""
    if not a or not b:
        return []
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    result = []
    for _ in range(rows_a):
        result.append([0.0] * cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0.0
            for k in range(cols_a):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result


def mandelbrot(cr: float, ci: float, max_iter: int) -> int:
    """Compute Mandelbrot escape iteration count for a point (cr, ci)."""
    if cr * cr + ci * ci >= 4.0:
        return 0
    zr = 0.0
    zi = 0.0
    for i in range(max_iter):
        new_zr = zr * zr - zi * zi + cr
        new_zi = 2.0 * zr * zi + ci
        if new_zr * new_zr + new_zi * new_zi > 4.0:
            return i + 1
        zr = new_zr
        zi = new_zi
    return max_iter


def timsort(arr: List[int]) -> List[int]:
    """Sort a list of numbers using Python's sorted (Timsort)."""
    return sorted(arr)


def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors using a single iterative loop."""
    result: float = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def cooley_tukey(real: List[float], imag: List[float]) -> (List[float], List[float]):
    """Iterative Cooley-Tukey FFT (in-place) using separate real/imag lists.
    Input length must be a power of two.
    """
    n = len(real)
    if n == 0:
        return ([], [])
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            (real[i], real[j]) = (real[j], real[i])
            (imag[i], imag[j]) = (imag[j], imag[i])
    length = 2
    while length <= n:
        half = length // 2
        angle = -2.0 * math.pi / length
        for start in range(0, n, length):
            for k in range(half):
                idx_even = start + k
                idx_odd = idx_even + half
                wk = angle * k
                cos_wk = math.cos(wk)
                sin_wk = math.sin(wk)
                re_even = real[idx_even]
                im_even = imag[idx_even]
                re_odd = real[idx_odd]
                im_odd = imag[idx_odd]
                re_t = re_odd * cos_wk - im_odd * sin_wk
                im_t = re_odd * sin_wk + im_odd * cos_wk
                real[idx_even] = re_even + re_t
                imag[idx_even] = im_even + im_t
                real[idx_odd] = re_even - re_t
                imag[idx_odd] = im_even - im_t
        length <<= 1
    return (real, imag)


def most_frequent(arr: list[int]) -> int:
    """Return the most frequent element in a list, or -1 if empty."""
    n = len(arr)
    if n == 0:
        return -1
    sorted_arr = sorted(arr)
    max_count = 0
    most_freq = sorted_arr[0]
    current_count = 1
    for i in range(1, n):
        if sorted_arr[i] == sorted_arr[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                most_freq = sorted_arr[i - 1]
            current_count = 1
    if current_count > max_count:
        most_freq = sorted_arr[n - 1]
    return most_freq
