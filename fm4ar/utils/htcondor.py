"""
Methods for dealing with the HTCondor cluster system.
"""

import socket
import sys
from pathlib import Path
from subprocess import run
from typing import Literal

from pydantic import AliasChoices, BaseModel, Extra, Field


class HTCondorConfig(BaseModel):
    """
    Dataclass for storing the settings for an HTCondor job.
    """

    # Don't allow any extra fields not specified here
    # The point of this is to catch errors like a config file specifying a
    # value for `memory_cpu` (instead of `memory_cpus`), and this typo then
    # being silently ignored and the model using the default for `memory_cpus`
    class Config:
        extra = Extra.forbid

    executable: str = Field(
        default=sys.executable,
        description="Path to the executable (e.g., the Python interpreter).",
    )
    getenv: bool = Field(
        default=True,
        description="Whether the environment variables should be copied.",
    )
    gpu_type: Literal["A100", "H100"] | None = Field(
        default=None,  # don't request a specific GPU type
        description="Type of GPU to request for the job.",
    )
    n_cpus: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("n_cpus", "num_cpus"),
        description="Number of CPUs to request for the job.",
    )
    memory_cpus: int = Field(
        default=4_096,
        ge=1024,  # 1 GB is the minimum unit of memory on the cluster
        description="Amount of memory (in MB) to request for the job.",
    )
    n_gpus: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("n_gpus", "num_gpus"),
        description="Number of GPUs to request for the job.",
    )
    memory_gpus: int = Field(
        default=15_000,
        ge=0,
        description="Amount of GPU memory (in MB) to request for the job.",
    )
    arguments: str | list[str] = Field(
        default="",
        description="(List of) arguments to pass to the executable.",
    )
    requirements: list[str] = Field(
        default=[],
        description=(
            "Any additional requirements for the job, e.g., constraints "
            "on the nodes on which the job can run."
        ),
    )
    retry_on_exit_code: int | None = Field(
        default=None,
        description=(
            "If the job exits with this exit code, keep retrying. "
            "If `None`, do not retry."
        ),
    )
    log_file_name: str = Field(
        default="log",
        description="Base name for the log files.",
    )
    queue: int = Field(
        default=1,
        ge=1,
        description="Number of times top place this job in the queue.",
    )
    bid: int = Field(
        default=50,
        ge=15,  # 15 is the current minimum bid on the cluster
        description="Bid to use for the job.",
    )
    extra_kwargs: dict = Field(
        default={},
        description=(
            "Extra key/value pairs to add to the submission file. "
            "Example: transfer_executable = False."
        ),
    )


class DAGManFile:
    """
    Wrapper to create a DAGMan file.

    Note: This is pretty basic and does currently not implement more
    advanced  features likes topological sorting or cycle detection.
    It is assumed that the user knows what they are doing.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def add_job(
        self,
        name: str,
        file_path: Path,
        bid: int = 15,
        depends_on: list[str] | None = None,
    ) -> None:
        """
        Add job to the DAGMan file.

        Args:
            name: Name of the job (must be unique).
            file_path: Path to the submission file.
            bid: Bid to use for the job (default: 15).
            depends_on: Names of the parent jobs (i.e., jobs that need
                to finish before the current job can be launched).
        """

        # Make sure the job does not exist already
        if name in self.jobs:
            raise ValueError(f"Job '{name}' already exists!")

        # Add the job to the DAGman file
        self.jobs[name] = {
            "file_path": file_path,
            "bid": bid,
            "depends_on": [] if depends_on is None else depends_on,
        }

    def remove_job(self, name: str) -> None:
        """
        Remove a job from the DAGMan file.
        Note: This leaves any dependencies untouched.

        Args:
            name: Name of the job to remove.
        """

        if name not in self.jobs:
            raise ValueError(f"Job '{name}' does not exist!")

        del self.jobs[name]

    def save(self, file_path: Path) -> None:
        """
        Save the DAGMan file to disk.
        """

        # Collect lines. First, define the jobs:
        lines = []
        for name, job in self.jobs.items():
            lines.append(f"JOB {name} {job['file_path']}\n")
        lines.append("\n")

        # Then add the dependencies:
        for name, job in self.jobs.items():
            for parent in job["depends_on"]:
                if parent not in self.jobs:
                    raise ValueError(f"Parent '{parent}' does not exist!")
                lines.append(f"PARENT {parent} CHILD {name}\n")
        lines.append("\n")

        # Finally, add the bids (which we need to convert to priorities):
        for name, job in self.jobs.items():
            priority = job["bid"] - 1000
            lines.append(f"PRIORITY {name} {priority}\n")

        # Write the DAGman file (we do this at the end to make sure that we
        # don't write the file if there is an error)
        with open(file_path, "w") as f:
            for line in lines:
                f.write(line)


def check_if_on_login_node(start_submission: bool) -> None:
    """
    Check if this script is running on the login node of the cluster,
    and it's not only to start a submission. If yes, exit with an error.
    """

    if "login" in socket.gethostname() and not start_submission:
        print("Did you forget to add the `--start-submission` flag again?\n")
        sys.exit(1)


def condor_submit_bid(
    file_path: Path,
    bid: int = 50,
    verbose: bool = True,
) -> None:  # pragma: no cover
    """
    Submit a job to HTCondor using the bid system.

    Note: This function is marked as `no cover` because there is no
    meaningful way to test it.

    Args:
        file_path: Path to the submission file.
        bid: Bid to use for the job (default: 15).
        verbose: If True, print the output of `condor_submit_bid`.
    """

    cmd = ["condor_submit_bid", str(bid), str(file_path)]
    process = run(cmd, capture_output=True, check=False)

    if verbose:
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))


def condor_submit_dag(
    file_path: Path,
    verbose: bool = True,
    bid: int = 50,
) -> None:  # pragma: no cover
    """
    Submit a DAGMan file to HTCondor.

    Args:
        file_path: Path to the DAGMan file.
        verbose: If True, print the output of `condor_submit_dag`.
        bid: Bid to use for the job (default: 25).
    """

    cmd = ["condor_submit_dag_bid", str(bid), str(file_path)]
    process = run(cmd, capture_output=True, check=False)

    if verbose:
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))


def create_submission_file(
    htcondor_config: HTCondorConfig,
    experiment_dir: Path,
    file_name: str = "run.sub",
) -> Path:
    """
    Create a new submission file for HTCondor.

    Args:
        htcondor_config: A `HTConcodorConfig` object containing the
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
    lines.append(f"executable = {htcondor_config.executable}\n")
    lines.append(f"getenv = {htcondor_config.getenv}\n\n")

    # CPUs and memory requirements
    lines.append(f"request_cpus = {htcondor_config.n_cpus}\n")
    lines.append(f"request_memory = {htcondor_config.memory_cpus}\n")

    # Collect general requirements (e.g., constraints on the nodes)
    requirements = htcondor_config.requirements.copy()

    # Add GPU requirements
    if htcondor_config.n_gpus > 0:

        # Request the desired number of GPUs
        lines.append(f"request_gpus = {htcondor_config.n_gpus}\n")

        # Construct other requirements: GPU memory and / or type
        if (memory_gpus := htcondor_config.memory_gpus) > 0:
            requirements.append(f"TARGET.CUDAGlobalMemoryMb > {memory_gpus}")
        if (gpu_type := htcondor_config.gpu_type) is not None:
            cuda_capability = get_cuda_capability(gpu_type)
            requirements.append(f"TARGET.CUDACapability == {cuda_capability}")

    # Add the combined requirements to the submission file
    if requirements:
        lines.append(f"requirements = ({' && '.join(requirements)})\n\n")

    # Set the arguments
    arguments = (
        " ".join(htcondor_config.arguments)
        if isinstance(htcondor_config.arguments, list)
        else htcondor_config.arguments
    )
    lines.append(f'arguments = "{arguments}"\n\n')

    # Set up the log files
    name = htcondor_config.log_file_name
    lines.append(f'error = {logs_dir / f"{name}.err"}\n')
    lines.append(f'output = {logs_dir / f"{name}.out"}\n')
    lines.append(f'log = {logs_dir / f"{name}.log"}\n\n')

    # If get get a particular exit code, keep retrying
    if (exit_code := htcondor_config.retry_on_exit_code) is not None:
        lines.append(f"on_exit_hold = (ExitCode =?= {exit_code})\n")
        lines.append('on_exit_hold_reason = "Checkpointed, will resume"\n')
        lines.append("on_exit_hold_subcode = 1\n")
        lines.append(
            "periodic_release = ( (JobStatus =?= 5) "
            "&& (HoldReasonCode =?= 3) "
            "&& (HoldReasonSubCode =?= 1) )\n\n"
        )

    # Add extra key/value pairs, if applicable
    if htcondor_config.extra_kwargs:
        for key, value in htcondor_config.extra_kwargs.items():
            lines.append(f"{key} = {value}\n")
        lines.append("\n")

    # Set the queue
    queue = "" if htcondor_config.queue == 1 else htcondor_config.queue
    lines.append(f"queue {queue}")

    # Write the submission file
    file_path = experiment_dir / file_name
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)

    return file_path


def get_cuda_capability(gpu_type: Literal["A100", "H100"] | None) -> float:
    """
    Get the CUDA capability of the given GPU type.
    """

    match gpu_type:
        case "H100":
            return 9.0
        case "A100":
            return 8.0
        case None:
            return 1.0
        case _:
            raise ValueError(f"Unknown GPU type: {gpu_type}")
