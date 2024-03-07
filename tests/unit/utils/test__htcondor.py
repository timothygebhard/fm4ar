"""
Unit tests for `fm4ar.utils.htcondor`.
"""

import socket
from pathlib import Path

import pytest

from fm4ar.utils.htcondor import (
    CondorSettings,
    DAGManFile,
    check_if_on_login_node,
    create_submission_file,
)


def test__check_if_on_login_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `check_if_on_login_node()`.
    """

    # noinspection PyArgumentList
    with monkeypatch.context() as context:
        context.setattr(socket, "gethostname", lambda: "login")
        with pytest.raises(SystemExit) as excinfo:
            check_if_on_login_node(False)
        assert excinfo.value.code == 1


def test__create_submission_file(tmp_path: Path) -> None:
    """
    Test `create_submission_file()`.
    """

    # Case 1: Invalid experiment directory
    with pytest.raises(FileNotFoundError) as file_not_found_error:
        create_submission_file(
            condor_settings=CondorSettings(arguments=""),
            experiment_dir=tmp_path / "does_not_exist",
        )
    assert "Experiment directory does not exist" in str(file_not_found_error)

    # Case 2: Arguments as string
    file_path = create_submission_file(
        condor_settings=CondorSettings(
            arguments="arguments as string",
            num_gpus=1,
            gpu_type="A100",
        ),
        experiment_dir=tmp_path,
        file_name="run.sub",
    )
    assert file_path.exists()

    # Case 3: Arguments as list; retry_on_exit_code; extra kwargs
    file_path = create_submission_file(
        condor_settings=CondorSettings(
            arguments=["arguments", "as", "list"],
            num_gpus=1,
            retry_on_exit_code=42,
            extra_kwargs={"transfer_excecutable": "False"},
        ),
        experiment_dir=tmp_path,
        file_name="run.sub",
    )
    assert file_path.exists()


def test__dagman_file(tmp_path: Path) -> None:
    """
    Test `DAGManFile`.
    """

    dagman_file = DAGManFile()

    # Case 1: Add two jobs with a dependency
    dagman_file.add_job(
        name="job1",
        file_path=tmp_path / "job1.sub",
        bid=15,
        depends_on=None,
    )
    dagman_file.add_job(
        name="job2",
        file_path=tmp_path / "job2.sub",
        bid=15,
        depends_on=["job1"],
    )
    dagman_file.save(tmp_path / "dagman.dag")
    assert (tmp_path / "dagman.dag").exists()

    # Case 2: Add a job with a dependency on a non-existing job
    # Note: This only throws an error when saving the DAGMan file,
    # because we might add jobs in a non-sequential order.
    dagman_file.add_job(
        name="job3",
        file_path=tmp_path / "job3.sub",
        bid=15,
        depends_on=["job4"],
    )
    with pytest.raises(ValueError) as value_error:
        dagman_file.save(tmp_path / "dagman.dag")
    assert "Parent 'job4' does not exist" in str(value_error)

    # Case 3: Remove a job 3 that does exist
    dagman_file.remove_job("job3")
    assert "job4" not in dagman_file.jobs

    # Case 4: Remove a job 4 that does not exist
    with pytest.raises(ValueError) as value_error:
        dagman_file.remove_job("job4")
    assert "Job 'job4' does not exist" in str(value_error)

    # Case 5: Add the same job twice
    with pytest.raises(ValueError) as value_error:
        dagman_file.add_job(name="job1", file_path=tmp_path / "job1.sub")
    assert "Job 'job1' already exists!" in str(value_error)
