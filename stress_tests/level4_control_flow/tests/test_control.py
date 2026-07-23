from break_continue import find_first_factor, sum_skipping_multiples


def test_find_first_factor():
    assert find_first_factor(9) == 3
    assert find_first_factor(13) == -1


def test_sum_skipping_multiples():
    assert sum_skipping_multiples(10) == 27  # 1+2+4+5+7+8
