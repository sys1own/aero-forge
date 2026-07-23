def find_first_factor(n):
    for i in range(2, n):
        if n % i == 0:
            return i
    return -1

def sum_skipping_multiples(limit):
    total = 0
    for i in range(1, limit):
        if i % 3 == 0:
            continue
        total += i
    return total
