class Counter:
    def __init__(self, start: int):
        self.count = start

    def increment(self) -> int:
        self.count += 1
        return self.count

    def get(self) -> int:
        return self.count
