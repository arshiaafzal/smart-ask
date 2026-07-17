import unittest
from impl import get_module_docstring, rewrite_assertions


class TestModuleDocstring(unittest.TestCase):

    def test_string_constant_is_docstring(self):
        """A leading string constant is recognized as the docstring."""
        import ast
        tree = ast.parse('"""Hello world."""\nx = 1')
        doc = get_module_docstring(tree)
        self.assertEqual(doc, "Hello world.")

    def test_number_constant_is_not_docstring(self):
        """A leading integer constant must NOT be treated as a docstring."""
        import ast
        tree = ast.parse("42\nx = 1")
        doc = get_module_docstring(tree)
        self.assertIsNone(
            doc,
            "Integer literal at module start must not be treated as docstring",
        )

    def test_empty_module_returns_none(self):
        """Empty module has no docstring."""
        import ast
        tree = ast.parse("")
        self.assertIsNone(get_module_docstring(tree))

    def test_rewrite_with_number_first_no_crash(self):
        """rewrite_assertions must not crash when the first stmt is a number."""
        try:
            result = rewrite_assertions("42\nx = 1")
        except (IndexError, TypeError) as exc:
            self.fail(f"rewrite_assertions crashed on numeric first stmt: {exc}")
        # The number literal is NOT removed (it's not a docstring).
        self.assertIn("42", result)

    def test_rewrite_removes_real_docstring(self):
        """rewrite_assertions removes a genuine string docstring."""
        result = rewrite_assertions('"""Module doc."""\nx = 1')
        self.assertNotIn("Module doc", result)
        self.assertIn("x = 1", result)

    def test_float_constant_is_not_docstring(self):
        """A float constant at the start is also not a docstring."""
        import ast
        tree = ast.parse("3.14\ny = 2")
        doc = get_module_docstring(tree)
        self.assertIsNone(doc)


if __name__ == "__main__":
    unittest.main()
