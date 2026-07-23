def timsort(arr: list[int]) -> list[int]:
    RUN = 32
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
            temp = []
            i = left
            j = mid
            while i < mid and j < right:
                if arr[i] <= arr[j]:
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            while i < mid:
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            for k in range(left, right):
                arr[k] = temp[k - left]
        size *= 2
    return arr
