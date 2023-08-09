"""
Unit tests for `fm4ar.utils.htcondor`.
"""

import socket
from pathlib import Path

import pytest

from fm4ar.utils.htcondor import (
    check_if_on_login_node,
    copy_logfiles,
    create_submission_file,
)


def test__check_if_on_login_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `fm4ar.utils.htcondor.check_if_on_login_node()`.
    """

    with monkeypatch.context() as context:
        context.setattr(socket, "gethostname", lambda: "login")
        with pytest.raises(SystemExit) as excinfo:
            check_if_on_login_node(False)
        assert excinfo.value.code == 1


def test__copy_logfiles(tmp_path: Path) -> None:
    """
    Test `fm4ar.utils.htcondor.copy_logfiles()`.
    """

    # Create some dummy log files
    for suffix in ("log", "err", "out"):
        (tmp_path / f"info.42.{suffix}").touch()

    # Copy the log files
    copy_logfiles(log_dir=tmp_path, epoch=3)

    # Check that the log files have been copied
    for suffix in ("log", "err", "out"):
        assert (tmp_path / f"info.42.epoch-003.{suffix}").exists()

    # Check that files do not get backed up twice
    copy_logfiles(log_dir=tmp_path, epoch=3)
    assert len(list(tmp_path.glob("info.*"))) == 6


def test__create_submission_file(tmp_path: Path) -> None:
    """
    Test `fm4ar.utils.htcondor.create_submission_file()`.
    """

    # Case 1: Invalid experiment directory
    with pytest.raises(FileNotFoundError) as file_not_found_error:
        create_submission_file(
            condor_settings={"arguments": ""},
            experiment_dir=tmp_path / "does_not_exist",
        )
    assert "Experiment directory does not exist" in str(file_not_found_error)

    # Case 2: Arguments as string
    file_path = create_submission_file(
        condor_settings={"arguments": "arguments as string"},
        experiment_dir=tmp_path,
        file_name="run.sub",
    )
    assert file_path.exists()

    # Case 3: Arguments as list
    file_path = create_submission_file(
        condor_settings={
            "arguments": ["arguments", "as", "list"],
            "num_gpus": 1,
        },
        experiment_dir=tmp_path,
        file_name="run.sub",
    )
    assert file_path.exists()
