"""
Unit tests for `fm4ar.utils.environ`.
"""

import sys
from pathlib import Path

import pytest

from fm4ar.utils.environment import (
    document_environment,
    get_packages,
    get_python_version,
    get_virtual_environment,
)


def test__get_packages() -> None:
    """
    Unit test for `fm4ar.utils.environment.get_packages`.
    """

    packages = get_packages()
    assert len(packages) > 0
    assert any("fm4ar" in p for p in packages)


def test__get_python_version() -> None:
    """
    Unit test for `fm4ar.utils.environment.get_python_version`.
    """

    python_version = get_python_version()
    assert python_version == sys.version


def test__get_virtual_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Unit test for `fm4ar.utils.environment.get_virtual_environment`.
    """

    with monkeypatch.context() as mp:

        # Case 1: Virtual environment present
        mp.setenv("VIRTUAL_ENV", "my_virtual_environment")
        assert get_virtual_environment() == "my_virtual_environment"

        # Case 2: Virtual environment not present
        mp.delenv("VIRTUAL_ENV", raising=False)
        assert get_virtual_environment() == "No virtual environment detected"


def test__document_environment(tmp_path: Path) -> None:
    """
    Unit test for `fm4ar.utils.environment.document_environment`.
    """

    document_environment(target_dir=tmp_path)
    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / "requirements.txt").stat().st_size > 0
    with open(tmp_path / "requirements.txt", "r") as f:
        lines = f.readlines()
    assert any("fm4ar" in line for line in lines)
