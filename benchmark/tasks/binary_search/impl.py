"""
Binary search — two bugs planted.
"""


def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted arr, or -1 if not found."""
    lo, hi = 0, len(arr)          # Bug 1: hi should be len(arr) - 1
    while lo <= hi:
        mid = lo + hi // 2        # Bug 2: operator precedence — should be (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
