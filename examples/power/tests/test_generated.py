from generated import power


def test_power():
    assert power(2.0, 10) == 1024.0
    assert power(3.0, 0) == 1.0
    assert power(5.0, 3) == 125.0
