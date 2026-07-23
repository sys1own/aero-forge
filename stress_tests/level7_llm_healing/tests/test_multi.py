from multi import add_one, broken_double, subtract_one

def test_multi():
    assert add_one(5) == 6
    assert broken_double(4) == 8
    assert subtract_one(10) == 9
