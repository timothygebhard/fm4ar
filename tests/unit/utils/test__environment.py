"""
Unit tests for `fm4ar.utils.environ`.
"""

import sys
from pathlib import Path

from fm4ar.utils.environment import (
    get_packages,
    get_python_version,
    get_virtual_environment,
    document_environment,
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


def test__get_virtual_environment() -> None:
    """
    Unit test for `fm4ar.utils.environment.get_virtual_environment`.
    """

    virtual_environment = get_virtual_environment()
    assert isinstance(virtual_environment, str)


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
