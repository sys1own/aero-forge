METADATA = {
    "name": "fibonacci",
    "category": "math",
    "complexity": {"time": "O(2^n) naive, O(n) iterative", "space": "O(1)"},
    "use_cases": ["number theory", "generating sequences"],
    "constraints": ["naive recursion is slow; prefer iterative or memoized"],
}


def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
