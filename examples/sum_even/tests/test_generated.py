from generated import sum_even


def test_sum_even():
    assert sum_even(0) == 0
    assert sum_even(4) == 6
    assert sum_even(10) == 30
