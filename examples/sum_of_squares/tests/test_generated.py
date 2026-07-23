from generated import sum_of_squares


def test_sum_of_squares():
    assert sum_of_squares(0) == 0
    assert sum_of_squares(3) == 14
    assert sum_of_squares(10) == 385
