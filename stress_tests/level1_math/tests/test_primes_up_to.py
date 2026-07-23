from primes_up_to import primes_up_to


def test_primes_up_to():
    assert primes_up_to(10) == [2, 3, 5, 7]
    assert primes_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert primes_up_to(1) == []
