from matrix import Matrix


def test_matrix_multiply():
    a = Matrix(2, 3)
    a.data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = Matrix(3, 2)
    b.data = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    c = a.multiply(b)
    assert c.rows == 2
    assert c.cols == 2
    assert c.data == [[58.0, 64.0], [139.0, 154.0]]


def test_matrix_transpose():
    m = Matrix(2, 3)
    m.data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert m.transpose() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_matrix_get():
    m = Matrix(2, 2)
    m.data = [[1.0, 2.0], [3.0, 4.0]]
    assert m.get(0, 1) == 2.0
    assert m.get(1, 0) == 3.0
