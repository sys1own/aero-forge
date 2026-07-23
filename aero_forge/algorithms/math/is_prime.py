METADATA = {
    "name": "is_prime",
    "category": "math",
    "complexity": {"time": "O(sqrt(n))", "space": "O(1)"},
    "use_cases": ["primality testing", "cryptography basics"],
    "constraints": ["deterministic only for small numbers"],
}


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
