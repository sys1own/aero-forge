from matrix_multiply_append import matrix_multiply_append


def test_matrix_multiply_append():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    assert matrix_multiply_append(a, b) == [[19.0, 22.0], [43.0, 50.0]]
