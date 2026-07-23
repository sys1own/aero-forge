class Calculator:
    def __init__(self, start):
        self.value = start

    def add(self, x):
        self.value += x
        return self.value

    def get_value(self):
        return self.value

    @staticmethod
    def static_sum(a, b):
        return a + b

    @classmethod
    def from_value(cls, value):
        return cls(value)
