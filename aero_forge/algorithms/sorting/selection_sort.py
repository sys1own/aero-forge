from typing import List

METADATA = {
    "name": "selection_sort",
    "category": "sorting",
    "complexity": {"time": "O(n^2)", "space": "O(1)"},
    "use_cases": ["educational", "small arrays", "minimizing swaps"],
    "constraints": ["not stable", "poor performance on large inputs"],
}


def selection_sort(arr: List[int]) -> List[int]:
    arr = arr[:]
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
