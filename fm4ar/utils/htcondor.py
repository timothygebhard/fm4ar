"""
Methods for dealing with the HTCondor cluster system.
"""

from dataclasses import dataclass

import socket
import sys

from pathlib import Path
from shutil import copyfile
from subprocess import run


@dataclass
class CondorSettings:
    """
    Dataclass for storing the settings for an HTCondor job.
    """

    executable: str = sys.executable
    getenv: bool = True
    num_cpus: int = 1
    memory_cpus: int = 4_096
    num_gpus: int = 0
    memory_gpus: int = 15_000
    arguments: str | list[str] = ""
    # max_runtime: int | None = None
    # max_retries: int | None = None
    retry_on_exit_code: int | None = None
    log_file_name: str = "info"
    queue: int = 1
    bid: int = 15


def check_if_on_login_node(start_submission: bool) -> None:
    """
    Check if this script is running on the login node of the cluster,
    and it's not only to start a submission. If yes, exit with an error.
    """

    if "login" in socket.gethostname() and not start_submission:
        print("Did you forget to add the `--start-submission` flag again?\n")
        sys.exit(1)


def copy_logfiles(log_dir: Path, label: str) -> None:
    """
    Copy the log files to a new file with the epoch number appended.

    Args:
        log_dir: Path to the directory containing the log files.
        label: Label to add to the file name, e.g., the epoch number,
            or the number of the HTCondor job.
    """

    # Loop over all log files in the directory
    # Their names should follow the pattern `info.<Process>.{log,err,out}`.
    # Backup files are named `info.<Process>.<label>.{log,err,out}`.
    for src in log_dir.glob("info.*"):

        # Skip files that already have been copied before
        parts = src.name.split(".")
        if len(parts) > 3:
            continue

        # Copy the file to a new file with the epoch number appended
        name = ".".join(parts[:-1]) + f".{label}." + parts[-1]
        dst = log_dir / name
        try:
            copyfile(src, dst)
        except Exception as e:  # pragma: no cover
            print(f"Failed to copy file {src} to {dst}: {e}")


def condor_submit_bid(
    file_path: Path,
    bid: int = 15,
) -> None:  # pragma: no cover
    """
    Submit a job to HTCondor using the bid system.

    Note: This function is marked as `no cover` because there is no
    meaningful way to test it.

    Args:
        file_path: Path to the submission file.
        bid: Bid to use for the job (default: 15).
    """

    cmd = ["condor_submit_bid", str(bid), str(file_path)]
    run(cmd, capture_output=True, check=True)


def create_submission_file(
    condor_settings: CondorSettings,
    experiment_dir: Path,
    file_name: str = "run.sub",
) -> Path:
    """
    Create a new submission file for HTCondor.

    Args:
        condor_settings: A `CondorSettings` object containing the
            settings for the HTCondor job.
        experiment_dir: Path to the experiment directory where the
            submission file will be created.
        file_name: Name to use for the submission file.

    Returns:
        Path to the submission file that was created.
    """

    # Ensure that the experiment directory exists
    if not experiment_dir.exists():
        raise FileNotFoundError("Experiment directory does not exist!")

    # Ensure that the logs subdirectory exists
    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Collect contents of the submission file (with reasonable defaults)
    lines = []

    # Executable and environment variables
    lines.append(f'executable = {condor_settings.executable}\n')
    lines.append(f"getenv = {condor_settings.getenv}\n\n")

    # CPUs and memory requirements
    lines.append(f'request_cpus = {condor_settings.num_cpus}\n')
    lines.append(f'request_memory = {condor_settings.memory_cpus}\n')

    # Set GPU requirements (only add this section if GPUs are requested)
    if condor_settings.num_gpus > 0:
        lines.append(f'request_gpus = {condor_settings.num_gpus}\n')
        lines.append(
            f"requirements = TARGET.CUDAGlobalMemoryMb "
            f"> {condor_settings.memory_gpus}\n\n"
        )

    # Set the arguments
    arguments = (
        " ".join(condor_settings.arguments)
        if isinstance(condor_settings.arguments, list)
        else condor_settings.arguments
    )
    lines.append(f'arguments = "{arguments}"\n\n')

    # Set up the log files
    name = condor_settings.log_file_name
    lines.append(f'error = {logs_dir / f"{name}.err"}\n')
    lines.append(f'output = {logs_dir / f"{name}.out"}\n')
    lines.append(f'log = {logs_dir / f"{name}.log"}\n\n')

    # If get get a particular exit code, keep retrying
    if (exit_code := condor_settings.retry_on_exit_code) is not None:
        lines.append(f"on_exit_hold = (ExitCode =?= {exit_code})\n")
        lines.append('on_exit_hold_reason = "Checkpointed, will resume"\n')
        lines.append("on_exit_hold_subcode = 1\n")
        lines.append(
            "periodic_release = ( (JobStatus =?= 5) "
            "&& (HoldReasonCode =?= 3) "
            "&& (HoldReasonSubCode =?= 1) )\n\n"
        )

    # Set the queue
    queue = "" if condor_settings.queue == 1 else condor_settings.queue
    lines.append(f"queue {queue}")

    # Write the submission file
    file_path = experiment_dir / file_name
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)

    return file_path
