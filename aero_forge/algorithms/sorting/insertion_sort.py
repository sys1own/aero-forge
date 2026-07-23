from typing import List

METADATA = {
    "name": "insertion_sort",
    "category": "sorting",
    "complexity": {"time": "O(n^2) worst, O(n) best", "space": "O(1)"},
    "use_cases": ["small arrays", "nearly sorted data", "online sorting"],
    "constraints": ["inefficient for large random arrays"],
}


def insertion_sort(arr: List[int]) -> List[int]:
    arr = arr[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
