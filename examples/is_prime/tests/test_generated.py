from generated import is_prime


def test_is_prime():
    assert is_prime(2) is True
    assert is_prime(17) is True
    assert is_prime(18) is False
