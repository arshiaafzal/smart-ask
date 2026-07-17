"""
Minimal reproduction of django FILE_UPLOAD_PERMISSIONS bug.
SWE-bench: django__django-10914

FILE_UPLOAD_PERMISSIONS has no default value (None), causing files saved
by the file system storage to get inconsistent permissions depending on
the upload handler used (memory: 0o600 via mkstemp; disk: system umask).
"""

import os
import tempfile


# Bug: FILE_UPLOAD_PERMISSIONS is None instead of 0o644
FILE_UPLOAD_PERMISSIONS = None   # should be 0o644
FILE_UPLOAD_TEMP_DIR = None


def save_uploaded_file(content: bytes, destination: str) -> int:
    """
    Save uploaded content to destination, applying FILE_UPLOAD_PERMISSIONS.

    Returns the permissions actually set on the file.
    """
    # Write via a temp file (mimics Django's FileSystemStorage behaviour).
    fd, tmp = tempfile.mkstemp(dir=FILE_UPLOAD_TEMP_DIR)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)

    os.replace(tmp, destination)

    if FILE_UPLOAD_PERMISSIONS is not None:
        os.chmod(destination, FILE_UPLOAD_PERMISSIONS)
        return FILE_UPLOAD_PERMISSIONS
    else:
        # No explicit permission — returns whatever mkstemp / umask gave.
        return os.stat(destination).st_mode & 0o777
