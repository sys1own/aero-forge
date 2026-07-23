from generated import Counter


def test_counter():
    c = Counter(0)
    assert c.increment() == 1
    assert c.increment() == 2
    assert c.get() == 2
