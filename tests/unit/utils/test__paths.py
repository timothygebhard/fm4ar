"""
Unit tests for `fm4ar.utils.paths`.
"""

from pathlib import Path

import pytest

from fm4ar.utils.paths import (
    get_path_from_environment_variable,
    get_datasets_dir,
    get_experiments_dir,
    get_root_dir,
    expand_env_variables_in_path,
)


def test__get_path_from_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test `fm4ar.utils.paths.get_path_from_environment_variable()`.
    """

    # Case 1: Environment variable is not set
    # noinspection PyArgumentList
    with monkeypatch.context() as context:
        context.delenv("DUMMY_ENV_VAR", raising=False)
        with pytest.raises(ValueError) as value_error:
            get_path_from_environment_variable("DUMMY_ENV_VAR")
        assert "$DUMMY_ENV_VAR is not set!" in str(value_error)

    # Case 2: Environment variable is set, but path does not exist
    # noinspection PyArgumentList
    with monkeypatch.context() as context:
        context.setenv("DUMMY_ENV_VAR", str(tmp_path / "does_not_exist"))
        with pytest.raises(ValueError) as value_error:
            get_path_from_environment_variable("DUMMY_ENV_VAR")
        assert "$DUMMY_ENV_VAR is set, but" in str(value_error)

    # Case 3: Environment variable is set and path exists
    # noinspection PyArgumentList
    with monkeypatch.context() as context:
        context.setenv("DUMMY_ENV_VAR", str(tmp_path))
        assert get_path_from_environment_variable("DUMMY_ENV_VAR") == tmp_path


def test__get_datasets_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `fm4ar.utils.paths.get_datasets_dir()`.
    """

    # Case 1
    monkeypatch.setenv("FM4AR_DATASETS_DIR", ".")
    assert isinstance(get_datasets_dir(), Path)
    assert get_datasets_dir().as_posix() == "."

    # Case 2
    monkeypatch.delenv("FM4AR_DATASETS_DIR", raising=False)
    with pytest.raises(ValueError) as value_error:
        get_datasets_dir()
    assert '$FM4AR_DATASETS_DIR is not set' in str(value_error)


def test__get_experiments_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `fm4ar.utils.paths.get_experiments_dir()`.
    """

    # Case 1
    monkeypatch.setenv("FM4AR_EXPERIMENTS_DIR", ".")
    assert isinstance(get_datasets_dir(), Path)
    assert get_experiments_dir().as_posix() == "."

    # Case 2
    monkeypatch.delenv("FM4AR_EXPERIMENTS_DIR", raising=False)
    with pytest.raises(ValueError) as value_error:
        get_experiments_dir()
    assert '$FM4AR_EXPERIMENTS_DIR is not set' in str(value_error)


def test__get_root_dir() -> None:
    """
    Test `fm4ar.utils.paths.get_root_dir()`.
    """

    assert isinstance(get_root_dir(), Path)
    assert get_root_dir().as_posix().endswith("fm4ar")


def test__expand_env_variables_in_path() -> None:
    """
    Test `fm4ar.utils.paths.resolve_env_variables_in_path()`.
    """

    # Case 1
    path = Path("$HOME")
    assert expand_env_variables_in_path(path).as_posix() == str(Path.home())
