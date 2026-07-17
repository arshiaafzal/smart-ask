"""
Minimal reproduction of sympy ccode sinc printing bug.
SWE-bench: sympy__sympy-11400

ccode(sinc(x)) generates a "// Not supported" comment instead of the correct
piecewise C expression: (x != 0) ? sin(x)/x : 1.
Also, relational operators are not handled by the C printer.
"""

from __future__ import annotations
import math


class Expr:
    """Minimal symbolic expression base."""
    pass


class Symbol(Expr):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class Relational(Expr):
    def __init__(self, lhs, rhs, op: str):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op   # "!=", "==", "<", "<=", ">", ">="


class Sinc(Expr):
    """sinc(x) = sin(x)/x for x≠0, 1 for x=0."""
    def __init__(self, arg):
        self.arg = arg


class Piecewise(Expr):
    """Piecewise(expr1, cond1, expr2, otherwise)."""
    def __init__(self, *pieces):
        # Each piece is (value, condition) with last being the default.
        self.pieces = pieces


class CCodPrinter:
    """Simplified C code printer."""

    def doprint(self, expr) -> str:
        return self._print(expr)

    def _print(self, expr) -> str:
        if isinstance(expr, Symbol):
            return expr.name
        if isinstance(expr, Relational):
            return self._print_Relational(expr)
        if isinstance(expr, Sinc):
            return self._print_Sinc(expr)
        if isinstance(expr, Piecewise):
            return self._print_Piecewise(expr)
        if isinstance(expr, (int, float)):
            return str(expr)
        # Bug: unknown expressions fall through to a comment.
        return f"/* Not supported in C: {type(expr).__name__} */"

    def _print_Relational(self, expr: Relational) -> str:
        # Bug: _print_Relational is not implemented — returns unsupported comment.
        return f"/* Not supported in C: Relational */"

    def _print_Sinc(self, expr: Sinc) -> str:
        # Bug: _print_Sinc is not implemented — returns unsupported comment.
        return f"/* Not supported in C: Sinc */"

    def _print_Piecewise(self, expr: Piecewise) -> str:
        pieces = list(expr.pieces)
        # Build ternary chain: cond ? val : (cond2 ? val2 : default)
        if len(pieces) == 1:
            return self._print(pieces[0][0])
        val = self._print(pieces[0][0])
        cond = self._print(pieces[0][1])
        rest = self._print_Piecewise(Piecewise(*pieces[1:]))
        return f"(({cond}) ? ({val}) : ({rest}))"


def ccode(expr) -> str:
    """Print a symbolic expression as C code."""
    return CCodPrinter().doprint(expr)
