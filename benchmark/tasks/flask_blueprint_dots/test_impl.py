import unittest
from impl import Blueprint


class TestBlueprintDotValidation(unittest.TestCase):

    def test_valid_name_accepted(self):
        """Blueprint with a simple name (no dots) must be created."""
        bp = Blueprint("auth", __name__)
        self.assertEqual(bp.name, "auth")

    def test_dotted_name_raises(self):
        """Blueprint name containing a dot must raise ValueError."""
        with self.assertRaises(ValueError, msg="Expected ValueError for dotted blueprint name"):
            Blueprint("auth.admin", __name__)

    def test_dotted_name_message(self):
        """Error message should mention dots are not allowed."""
        try:
            Blueprint("auth.admin", __name__)
            self.fail("Expected ValueError")
        except ValueError as exc:
            self.assertIn(".", str(exc).lower() + "dot", msg=str(exc))

    def test_endpoint_dots_still_raise(self):
        """Existing endpoint-dot validation must still work."""
        bp = Blueprint("auth", __name__)
        with self.assertRaises(ValueError):
            @bp.route("/login", endpoint="admin.login")
            def login():
                pass

    def test_nested_dot_name_raises(self):
        """Multiple dots also raise ValueError."""
        with self.assertRaises(ValueError):
            Blueprint("a.b.c", __name__)

    def test_trailing_dot_raises(self):
        """Trailing dot in name must raise ValueError."""
        with self.assertRaises(ValueError):
            Blueprint("auth.", __name__)


if __name__ == "__main__":
    unittest.main()
