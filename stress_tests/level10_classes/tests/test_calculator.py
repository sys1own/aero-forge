from calculator import Calculator


def test_instance_methods():
    c = Calculator(10)
    assert c.add(5) == 15
    assert c.get_value() == 15


def test_staticmethod():
    assert Calculator.static_sum(2, 3) == 5


def test_classmethod():
    c = Calculator.from_value(7)
    assert c.get_value() == 7
