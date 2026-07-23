from matrix_multiply import matrix_multiply

def test_matrix_multiply():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = matrix_multiply(a, b)
    assert result == [[19, 22], [43, 50]]
