from typing import List, Optional

METADATA = {
    "name": "binary_search",
    "category": "searching",
    "complexity": {"time": "O(log n)", "space": "O(1)"},
    "use_cases": ["sorted array lookup", "lower/upper bounds"],
    "constraints": ["requires sorted input"],
}


def binary_search(arr: List[int], target: int) -> Optional[int]:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None
