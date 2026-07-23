from math_funcs import hypotenuse, sine_angle, cos_angle


def test_hypotenuse():
    assert abs(hypotenuse(3.0, 4.0) - 5.0) < 1e-9


def test_sine():
    assert abs(sine_angle(30.0) - 0.5) < 1e-9


def test_cosine():
    assert abs(cos_angle(60.0) - 0.5) < 1e-9
