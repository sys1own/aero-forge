def outer(x):
    def inner(y):
        return y + 1
    return inner(x)
