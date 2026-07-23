class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self):
        return self.x + self.y


def create_point(x, y):
    return Point(x, y).distance()
