"""
Utility function for computing hashes of files.
"""

import hashlib
from pathlib import Path


def get_sha512sum(file_path: Path, buffer_size: int = 4096) -> str:
    """
    Compute the SHA-512 hash of a file (e.g., to uniquely identify it).
    This should match output of `sha512sum` command.
    """

    hash_md5 = hashlib.sha512()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(buffer_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
