import unittest
from impl import Symbol, Sinc, Relational, Piecewise, ccode


class TestCCodeSinc(unittest.TestCase):

    def test_sinc_no_unsupported_comment(self):
        """ccode(sinc(x)) must NOT produce a 'Not supported' comment."""
        x = Symbol("x")
        result = ccode(Sinc(x))
        self.assertNotIn("Not supported", result, f"Got: {result}")

    def test_sinc_contains_sin(self):
        """ccode(sinc(x)) must reference sin(x) in the output."""
        x = Symbol("x")
        result = ccode(Sinc(x))
        self.assertIn("sin", result, f"Expected sin in output, got: {result}")

    def test_sinc_handles_zero_case(self):
        """sinc(x) output must include a branch for x==0 returning 1."""
        x = Symbol("x")
        result = ccode(Sinc(x))
        self.assertIn("1", result, f"Expected '1' (zero case) in: {result}")

    def test_relational_no_unsupported_comment(self):
        """Relational expressions must produce actual C code, not a comment."""
        x = Symbol("x")
        rel = Relational(x, 0, "!=")
        result = ccode(rel)
        self.assertNotIn("Not supported", result, f"Got: {result}")

    def test_relational_neq(self):
        """x != 0 should produce C 'x != 0'."""
        x = Symbol("x")
        rel = Relational(x, 0, "!=")
        result = ccode(rel)
        self.assertIn("!=", result)
        self.assertIn("x", result)

    def test_sinc_is_valid_c_expression(self):
        """The sinc output must be a ternary expression, not a comment."""
        x = Symbol("x")
        result = ccode(Sinc(x))
        # Should be a ternary: (condition ? value : fallback)
        self.assertIn("?", result, f"Expected ternary in: {result}")


if __name__ == "__main__":
    unittest.main()
