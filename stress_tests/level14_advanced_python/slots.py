class WithSlots:
    __slots__ = ["x"]

    def __init__(self, x: int):
        self.x = x
