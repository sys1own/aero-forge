from typing import List

METADATA = {
    "name": "mergesort",
    "category": "sorting",
    "complexity": {"time": "O(n log n)", "space": "O(n)"},
    "use_cases": ["stable sorting", "linked lists", "external sorting"],
    "constraints": ["requires O(n) auxiliary space"],
}


def mergesort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return _merge(left, right)


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
