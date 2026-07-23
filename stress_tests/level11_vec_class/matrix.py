class Matrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    def multiply(self, other: "Matrix") -> "Matrix":
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    def transpose(self) -> list[list[float]]:
        return [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]

    def get(self, row: int, col: int) -> float:
        return self.data[row][col]
