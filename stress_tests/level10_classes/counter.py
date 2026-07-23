class Counter:
    def __init__(self, start):
        self.value = start

    def increment(self, step):
        self.value += step
        return self.value
