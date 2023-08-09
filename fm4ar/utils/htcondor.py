"""
Methods for dealing with the HTCondor cluster system.
"""

import socket
import sys

from pathlib import Path
from shutil import copyfile
from subprocess import run
from typing import Any


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
    condor_settings: dict[str, Any],
    experiment_dir: Path,
    file_name: str = "run.sub",
) -> Path:
    """
    Create a new submission file for HTCondor.

    Args:
        condor_settings: Dictionary containing the settings for the
            HTCondor job (e.g., number of CPUs, memory, etc.).
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

    # Unless otherwise specified, use the current executable
    executable = condor_settings.get("executable", sys.executable)
    lines.append(f'executable = {executable}\n')

    # Unless otherwise specified, copy environment variables
    getenv = condor_settings.get("getenv", True)
    lines.append(f"getenv = {getenv}\n\n")

    # Set number of CPUs
    n_cpus = int(condor_settings.get("num_cpus", 1))
    lines.append(f'request_cpus = {n_cpus}\n')

    # Set memory requirements (in MB)
    memory = int(condor_settings.get("memory_cpus", 4096))
    lines.append(f'request_memory = {memory}\n')

    # Set GPU requirements (only add this section if GPUs are requested)
    n_gpus = int(condor_settings.get("num_gpus", 0))
    cuda_memory = int(condor_settings.get("memory_gpus", 15_000))
    if n_gpus > 0:
        lines.append(f'request_gpus = {n_gpus}\n')
        lines.append(
            f"requirements = TARGET.CUDAGlobalMemoryMb > {cuda_memory}\n\n"
        )

    # Set the arguments
    # At least one argument must be provided (the script to be run)
    if isinstance(condor_settings["arguments"], list):
        arguments = " ".join(condor_settings["arguments"])
    else:
        arguments = condor_settings["arguments"]
    lines.append(f'arguments = "{arguments}"\n\n')

    lines.append(f'error = {logs_dir / "info.$(Process).err"}\n')
    lines.append(f'output = {logs_dir / "info.$(Process).out"}\n')
    lines.append(f'log = {logs_dir / "info.$(Process).log"}\n\n')

    queue = condor_settings.get("queue", 1)
    queue = "" if queue == 1 else queue
    lines.append(f"queue {queue}")

    # Write the submission file
    file_path = experiment_dir / file_name
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)

    return file_path
