import unittest
from impl import inherited_members_option, filter_members


class TestInheritedMembersOption(unittest.TestCase):

    def test_none_means_all(self):
        """No argument → include from all bases (True)."""
        result = inherited_members_option(None)
        self.assertIs(result, True)

    def test_single_class_name(self):
        """Single class name → {class_name}."""
        result = inherited_members_option("BaseModel")
        self.assertEqual(result, {"BaseModel"})

    def test_comma_separated_list(self):
        """Comma-separated list → set of stripped names."""
        result = inherited_members_option("BaseA, BaseB, BaseC")
        self.assertEqual(result, {"BaseA", "BaseB", "BaseC"})

    def test_comma_no_spaces(self):
        """Comma without spaces → set of names."""
        result = inherited_members_option("BaseA,BaseB")
        self.assertEqual(result, {"BaseA", "BaseB"})

    def test_trailing_comma_ignored(self):
        """Trailing comma should not produce an empty string entry."""
        result = inherited_members_option("BaseA,BaseB,")
        self.assertNotIn("", result)
        self.assertEqual(result, {"BaseA", "BaseB"})

    def test_filter_with_set(self):
        """filter_members keeps only members from allowed bases."""
        members = ["save", "delete", "clean", "validate"]
        bases = ["Model", "Model", "Form", "Form"]
        opt = inherited_members_option("Model")
        result = filter_members(members, bases, opt)
        self.assertEqual(result, ["save", "delete"])

    def test_filter_with_multi_set(self):
        """filter_members with multiple allowed bases."""
        members = ["save", "delete", "clean", "validate"]
        bases = ["Model", "Model", "Form", "Form"]
        opt = inherited_members_option("Model, Form")
        result = filter_members(members, bases, opt)
        self.assertEqual(result, ["save", "delete", "clean", "validate"])

    def test_filter_all(self):
        """True means include all members."""
        members = ["a", "b", "c"]
        bases = ["X", "Y", "Z"]
        result = filter_members(members, bases, True)
        self.assertEqual(result, members)


if __name__ == "__main__":
    unittest.main()
