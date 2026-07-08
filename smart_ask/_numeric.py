"""Shared finite-arithmetic guards for runtime and metric values."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
import math
from numbers import Real
from statistics import mean
from typing import Any


_RANGE_DIAGNOSTIC = "cannot be represented as a finite aggregate"


def is_finite_real(value: Any) -> bool:
    """Whether a non-boolean real value safely converts to a finite float."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def checked_fsum(values: Iterable[float], *, name: str) -> float:
    """Return the correctly rounded finite sum of numeric observations."""

    try:
        exact_result = sum((Fraction(value) for value in values), Fraction())
        result = float(exact_result)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}") from exc
    if not math.isfinite(result) or (exact_result and result == 0.0):
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}")
    return result


def checked_mean(values: Sequence[float], *, name: str) -> float | None:
    """Return a finite mean without overflowing a representable result."""

    if not values:
        return None
    try:
        result = float(mean(values))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}")
    return result


def checked_difference(left: float, right: float, *, name: str) -> float:
    """Subtract two finite observations and reject a non-finite result."""

    result = float(left) - float(right)
    if not math.isfinite(result):
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}")
    return result


def checked_product(
    left: int | float,
    right: int | float,
    *,
    name: str,
) -> float:
    """Multiply numeric observations without premature float overflow."""

    try:
        exact_result = Fraction(left) * Fraction(right)
        result = float(exact_result)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}") from exc
    if not math.isfinite(result) or (exact_result and result == 0.0):
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}")
    return result


def checked_ratio(
    numerator: int | float,
    denominator: int | float,
    *,
    name: str,
) -> float:
    """Divide numeric observations without premature float overflow."""

    try:
        exact_result = Fraction(numerator) / Fraction(denominator)
        result = float(exact_result)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}") from exc
    if not math.isfinite(result) or (exact_result and result == 0.0):
        raise ValueError(f"{name} {_RANGE_DIAGNOSTIC}")
    return result


__all__ = [
    "checked_difference",
    "checked_fsum",
    "checked_mean",
    "checked_product",
    "checked_ratio",
    "is_finite_real",
]
