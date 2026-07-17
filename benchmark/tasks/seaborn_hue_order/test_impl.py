import unittest
from impl import HueLookup


class TestHueLookup(unittest.TestCase):

    def _make_lookup(self, order=None):
        full = {"cat": "blue", "dog": "red", "fish": "green"}
        # Only include order items in lookup to simulate seaborn's filtering.
        if order is not None:
            lookup = {k: v for k, v in full.items() if k in order}
        else:
            lookup = full
        return HueLookup(lookup, order=order)

    def test_all_keys_present(self):
        """Normal case: all hue values are in lookup."""
        lu = self._make_lookup()
        result = lu.map_series(["cat", "dog", "fish"])
        self.assertEqual(result, ["blue", "red", "green"])

    def test_partial_hue_order_no_error(self):
        """pairplot with hue_order subset must not raise KeyError."""
        lu = self._make_lookup(order=["cat", "dog"])
        # "fish" is in the data but not in hue_order — must not crash.
        try:
            result = lu.map_series(["cat", "fish", "dog"])
        except KeyError as exc:
            self.fail(f"map_series raised KeyError for missing hue value: {exc}")
        # Values in order map correctly; values not in order return None.
        self.assertEqual(result[0], "blue")   # cat
        self.assertIsNone(result[1])           # fish not in order → None
        self.assertEqual(result[2], "red")    # dog

    def test_empty_order_returns_all_none(self):
        """If hue_order is empty, every lookup returns None."""
        lu = HueLookup({}, order=[])
        result = lu.map_series(["cat", "dog"])
        self.assertEqual(result, [None, None])

    def test_single_hue_value_in_order(self):
        """Only one hue value in order; others return None."""
        lu = self._make_lookup(order=["cat"])
        result = lu.map_series(["cat", "dog"])
        self.assertEqual(result[0], "blue")
        self.assertIsNone(result[1])


if __name__ == "__main__":
    unittest.main()
