from generated import mandelbrot


def test_mandelbrot():
    assert mandelbrot(0.0, 0.0, 100) == 100
    assert mandelbrot(2.0, 2.0, 100) < 100
