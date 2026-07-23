from tuple_unpack import swap, minmax


def test_swap():
    x, y = swap(1, 2)
    assert x == 2
    assert y == 1


def test_minmax():
    lo, hi = minmax(3, 1, 2)
    assert lo == 1
    assert hi == 3
