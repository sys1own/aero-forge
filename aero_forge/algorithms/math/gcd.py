METADATA = {
    "name": "gcd",
    "category": "math",
    "complexity": {"time": "O(log min(a, b))", "space": "O(1)"},
    "use_cases": ["number theory", "fraction reduction"],
    "constraints": [],
}


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
