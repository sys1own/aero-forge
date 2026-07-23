from gcd import gcd


def test_gcd():
    assert gcd(48, 18) == 6
    assert gcd(100, 25) == 25
    assert gcd(17, 13) == 1
