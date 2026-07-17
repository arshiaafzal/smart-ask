"""
Minimal reproduction of astropy separability_matrix bug.
SWE-bench: astropy__astropy-12907

The _cstack function incorrectly fills the bottom-right corner with 1
instead of the actual right matrix, breaking separability for nested models.

Pure-Python version (no numpy required).
"""

from __future__ import annotations


def zeros(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def _cstack(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """Stack two separability matrices column-wise for a compound model."""
    n_left_out = len(left)
    n_right_out = len(right)
    n_left_in = len(left[0]) if left else 0
    n_right_in = len(right[0]) if right else 0

    noutp = n_left_out + n_right_out
    ninp = n_left_in + n_right_in
    result = zeros(noutp, ninp)

    # Copy left submatrix into top-left corner.
    for i in range(n_left_out):
        for j in range(n_left_in):
            result[i][j] = left[i][j]

    # Bug: fills bottom-right corner with 1 instead of the actual right matrix.
    for i in range(n_right_out):
        for j in range(n_right_in):
            result[n_left_out + i][n_left_in + j] = 1   # should be right[i][j]

    return result


def separability_matrix(
    left_matrix: list[list[int]],
    right_matrix: list[list[int]],
) -> list[list[int]]:
    """Compute the separability matrix for two parallel-composed models."""
    return _cstack(left_matrix, right_matrix)
