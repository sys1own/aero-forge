def mandelbrot(c_re: float, c_im: float, max_iter: int) -> int:
    z_re = 0.0
    z_im = 0.0
    for i in range(max_iter):
        if z_re * z_re + z_im * z_im > 4.0:
            return i
        new_re = z_re * z_re - z_im * z_im + c_re
        z_im = 2.0 * z_re * z_im + c_im
        z_re = new_re
    return max_iter
