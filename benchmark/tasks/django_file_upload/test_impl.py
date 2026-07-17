import os
import tempfile
import unittest
from impl import save_uploaded_file, FILE_UPLOAD_PERMISSIONS


class TestFileUploadPermissions(unittest.TestCase):

    def test_default_permissions_defined(self):
        """FILE_UPLOAD_PERMISSIONS must have a non-None default."""
        self.assertIsNotNone(
            FILE_UPLOAD_PERMISSIONS,
            "FILE_UPLOAD_PERMISSIONS must not be None; default should be 0o644",
        )

    def test_default_permissions_value(self):
        """Default permissions should be 0o644 (world-readable, owner-writable)."""
        self.assertEqual(
            FILE_UPLOAD_PERMISSIONS,
            0o644,
            f"Expected 0o644, got {oct(FILE_UPLOAD_PERMISSIONS) if FILE_UPLOAD_PERMISSIONS else None}",
        )

    def test_saved_file_has_correct_permissions(self):
        """Uploaded file must receive exactly FILE_UPLOAD_PERMISSIONS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "upload.txt")
            actual = save_uploaded_file(b"hello", dest)
            self.assertEqual(
                actual,
                0o644,
                f"Expected 0o644 ({oct(0o644)}), got {oct(actual)}",
            )

    def test_saved_file_permissions_consistent(self):
        """Two consecutive uploads must produce identical permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = save_uploaded_file(b"a", os.path.join(tmpdir, "f1.txt"))
            p2 = save_uploaded_file(b"b", os.path.join(tmpdir, "f2.txt"))
            self.assertEqual(p1, p2, "Permissions must be consistent across uploads")


if __name__ == "__main__":
    unittest.main()
