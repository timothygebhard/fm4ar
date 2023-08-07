"""
Handle paths (e.g., to datasets or experiments directory).
"""

from os import getenv
from pathlib import Path

import fm4ar


def get_path_from_environment_variable(name: str) -> Path:
    """
    Resolve a path from an environment variable.
    """

    if (value := getenv(name, None)) is None:
        raise ValueError(f"${name} is not set!")

    if not Path(value).exists():
        raise ValueError(f"${name} is set, but `{value}` does not exist!")

    return Path(value)


def get_datasets_dir() -> Path:
    """
    Return the path to the datasets directory.
    """

    return get_path_from_environment_variable("FM4AR_DATASETS_DIR")


def get_experiments_dir() -> Path:
    """
    Return the path to the experiments directory.
    """

    return get_path_from_environment_variable("FM4AR_EXPERIMENTS_DIR")


def get_root_dir() -> Path:
    """
    Return the path to the root directory of the repository.
    """

    return Path(fm4ar.__file__).parents[1]
