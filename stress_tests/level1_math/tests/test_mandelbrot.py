from mandelbrot import mandelbrot_iterations

def test_mandelbrot():
    assert mandelbrot_iterations(0.0, 0.0, 100) == 100
    assert mandelbrot_iterations(2.0, 0.0, 100) < 100
