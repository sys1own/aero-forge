from generated import gcd


def test_gcd():
    assert gcd(48, 18) == 6
    assert gcd(100, 35) == 5
    assert gcd(7, 13) == 1
