def find_primes_tuple(n: int) -> tuple[list[int], int]:
    if n < 2:
        return [], 0
    primes = []
    count = 0
    for i in range(2, n + 1):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            count += 1
    return primes, count
