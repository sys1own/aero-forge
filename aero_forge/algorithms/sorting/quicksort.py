from typing import List

METADATA = {
    "name": "quicksort",
    "category": "sorting",
    "complexity": {"time": "O(n log n) average, O(n^2) worst", "space": "O(log n)"},
    "use_cases": ["general-purpose sorting", "in-place sorting"],
    "constraints": ["not stable", "worst-case quadratic"],
}


def quicksort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
