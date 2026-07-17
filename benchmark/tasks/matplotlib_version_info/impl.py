"""
Minimal reproduction of matplotlib __version_info__ bug.
SWE-bench: matplotlib__matplotlib-18869

matplotlib only exposes __version__ as a string, but not a comparable
__version_info__ tuple, making version comparisons awkward.
"""

from collections import namedtuple

__version__ = "3.4.2"

# Bug: __version_info__ is missing entirely.
# Should be a namedtuple comparable to sys.version_info.


def _parse_to_version_info(version_str: str):
    """Parse a PEP-440 version string into a comparable named tuple."""
    raise NotImplementedError("_parse_to_version_info is not implemented")
