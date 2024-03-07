"""
Unit tests for `fm4ar.utils.hashing`.
"""

from pathlib import Path

from fm4ar.utils.hashing import get_sha512sum


def test__get_sha512sum(tmp_path: Path) -> None:
    """
    Test `get_sha512sum()`.
    """

    # Create a dummy file
    file_path = tmp_path / "dummy.txt"
    with open(file_path, "w") as f:
        f.write("42\n")

    # Compute the SHA-512 hash
    assert get_sha512sum(file_path).startswith("65f61ced21494aeaa7f9f2")
