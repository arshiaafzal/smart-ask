"""
Minimal reproduction of sphinx inherited-members option bug.
SWE-bench: sphinx-doc__sphinx-10325

inherited_members_option only accepts a single class name (or True for all).
Users need a comma-separated list so they can filter inherited members from
specific bases rather than including everything.
"""

from __future__ import annotations


def inherited_members_option(arg: str | None) -> set[str] | bool:
    """
    Process the :inherited-members: autodoc option.

    Should accept:
      - ``None`` / no arg → True (include from all bases)
      - ``"BaseClass"``   → {"BaseClass"} (one class)
      - ``"A, B, C"``     → {"A", "B", "C"} (comma-separated list)  ← Bug: not supported

    Bug: only handles the single-class case; comma-separated lists are not
    parsed and the full string is returned as-is (or raises).
    """
    if arg is None:
        return True
    # Bug: treats the whole string as a single class name.
    # Should split on commas and strip whitespace.
    return {arg}   # e.g. "A, B" → {"A, B"} instead of {"A", "B"}


def filter_members(
    members: list[str],
    bases: list[str],
    inherited_option: set[str] | bool,
) -> list[str]:
    """
    Filter member names, keeping only those that come from allowed bases.

    ``inherited_option`` is the parsed result of inherited_members_option().
    True means include inherited members from all bases; a set means only
    from the named bases.
    """
    if inherited_option is True:
        return members  # include everything
    if not inherited_option:
        return []
    # Keep members whose base is in the option set.
    # (In real sphinx, this checks member.__qualname__; simplified here.)
    return [m for m, b in zip(members, bases) if b in inherited_option]
