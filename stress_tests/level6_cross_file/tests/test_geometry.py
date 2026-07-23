from geometry import circle_area


def test_geometry():
    assert abs(circle_area(1.0) - 3.141592653589793) < 1e-9
