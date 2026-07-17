import unittest
import impl


class TestVersionInfo(unittest.TestCase):

    def test_version_info_exists(self):
        """__version_info__ must be importable from the module."""
        self.assertTrue(
            hasattr(impl, "__version_info__"),
            "impl must expose __version_info__",
        )

    def test_version_info_is_tuple(self):
        """__version_info__ must be a tuple (or named tuple) of ints."""
        vi = impl.__version_info__
        self.assertIsInstance(vi, tuple)
        self.assertGreaterEqual(len(vi), 3)

    def test_version_info_values(self):
        """__version_info__ values must match __version__ string."""
        vi = impl.__version_info__
        self.assertEqual(vi.major, 3)
        self.assertEqual(vi.minor, 4)
        self.assertEqual(vi.micro, 2)

    def test_version_info_comparable(self):
        """> and < comparisons must work as with sys.version_info."""
        vi = impl.__version_info__
        self.assertGreater(vi, (3, 0, 0))
        self.assertLess(vi, (4, 0, 0))

    def test_parse_to_version_info_release_candidate(self):
        """RC versions parse correctly."""
        vi = impl._parse_to_version_info("3.5.0rc2")
        self.assertEqual(vi.major, 3)
        self.assertEqual(vi.minor, 5)
        self.assertEqual(vi.micro, 0)

    def test_parse_to_version_info_dev(self):
        """Dev/post versions parse without crashing."""
        vi = impl._parse_to_version_info("3.5.0.dev123")
        self.assertEqual(vi.major, 3)


if __name__ == "__main__":
    unittest.main()
