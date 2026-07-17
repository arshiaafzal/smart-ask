"""
Minimal reproduction of pytest assertion rewrite docstring detection bug.
SWE-bench: pytest-dev__pytest-11143

When the first statement in a module body is a number literal (e.g., `42`),
the assertion rewriter mistakenly treats it as a docstring, causing an
IndexError when it tries to remove it from the AST body.
"""

from __future__ import annotations
import ast


def get_module_docstring(tree: ast.Module) -> str | None:
    """
    Return the module-level docstring if the first statement is a string
    constant, otherwise return None.

    Bug: does not check that the constant is actually a *string* — a numeric
    literal at the top of the file triggers the same branch and then crashes
    when the rewriter tries to remove it.
    """
    if not tree.body:
        return None
    first = tree.body[0]
    # Bug: only checks isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    # but does NOT verify that the constant is a str.
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        # Returns the constant value regardless of type — should guard on str.
        return first.value.value   # crashes downstream for non-str constants
    return None


def rewrite_assertions(source: str) -> str:
    """
    Parse source and remove any detected docstring, returning the rewritten
    module body as a list of statement strings (simplified).

    Raises IndexError when the first statement is a non-string constant.
    """
    tree = ast.parse(source)
    doc = get_module_docstring(tree)
    if doc is not None:
        # Remove the docstring node from the body.
        tree.body.pop(0)
    return ast.unparse(tree)
