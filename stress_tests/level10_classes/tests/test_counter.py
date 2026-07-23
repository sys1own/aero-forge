from counter import Counter


def test_counter():
    c = Counter(10)
    assert c.increment(5) == 15
    assert c.increment(3) == 18
