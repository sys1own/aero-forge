from timsort import timsort


def test_timsort():
    assert timsort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
    assert timsort([]) == []
    assert timsort([5]) == [5]
    assert timsort([2, 2, 2]) == [2, 2, 2]
