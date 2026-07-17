"""
Minimal reproduction of seaborn pairplot hue_order KeyError bug.
SWE-bench: mwaskom__seaborn-2848

When hue_order contains only a subset of hue values, _lookup_single raises
a KeyError instead of returning a default/null value for missing keys.
"""

from __future__ import annotations
from typing import Any


class HueLookup:
    """Maps hue category values to plot attributes (color, marker, etc.)."""

    def __init__(self, lookup: dict[str, Any], order: list[str] | None = None):
        self._lookup = lookup
        self._order = order  # subset of keys to display

    def _lookup_single(self, key: str) -> Any:
        """Return the attribute for key, or None if key not in hue_order.

        Bug: raises KeyError instead of returning None for keys not in lookup.
        """
        # Bug: no guard for missing keys when order filters some values out.
        return self._lookup[key]   # should use .get(key) or handle KeyError

    def map_series(self, series: list[str]) -> list[Any]:
        """Map a list of category labels to their attributes."""
        return [self._lookup_single(v) for v in series]
