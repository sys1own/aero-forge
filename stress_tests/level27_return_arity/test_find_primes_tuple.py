from find_primes_tuple import find_primes_tuple


def test_find_primes_tuple():
    assert find_primes_tuple(1) == ([], 0)
    primes, count = find_primes_tuple(10)
    assert primes == [2, 3, 5, 7]
    assert count == 4
