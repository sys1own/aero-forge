from find_primes import find_primes


def test_find_primes():
    primes, count = find_primes(10)
    assert primes == [2, 3, 5, 7]
    assert count == 4
