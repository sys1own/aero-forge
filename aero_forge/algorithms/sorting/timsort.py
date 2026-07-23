from typing import List

METADATA = {
    "name": "timsort",
    "category": "sorting",
    "complexity": {"time": "O(n log n)", "space": "O(n)"},
    "use_cases": ["real-world data with runs", "Python built-in sort"],
    "constraints": ["complex implementation", "benefits from existing order"],
}


def timsort(arr: List[int]) -> List[int]:
    """Simplified reference: uses insertion sort for small runs and merge."""
    RUN = 32
    arr = arr[:]
    n = len(arr)
    for start in range(0, n, RUN):
        end = min(start + RUN, n)
        for i in range(start + 1, end):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    size = RUN
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(left + size, n)
            right = min(left + 2 * size, n)
            arr[left:right] = _merge(arr[left:mid], arr[mid:right])
        size *= 2
    return arr


def _merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
