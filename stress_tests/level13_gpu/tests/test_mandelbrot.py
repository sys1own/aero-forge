from mandelbrot import mandelbrot


def test_mandelbrot_inside():
    # The origin stays bounded, so it should reach max_iter.
    assert mandelbrot(0.0, 0.0, 100) == 100


def test_mandelbrot_outside():
    # (2, 2) diverges quickly.
    assert mandelbrot(2.0, 2.0, 100) < 100
