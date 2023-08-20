"""
Unified script to run different nested sampling algorithms.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from coolname import generate_slug

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, NAMES
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.nested_sampling import (
    create_posterior_plot,
    get_target_parameters_and_spectrum,
)
from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.utils.git_utils import document_git_status
from fm4ar.utils.htcondor import (
    CondorSettings,
    check_if_on_login_node,
    create_submission_file,
    condor_submit_bid,
)


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bid",
        type=int,
        default=25,
        help="Bid to use for the HTCondor job (default: 25).",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=4 * 60 * 60,
        help="Maximum runtime (in seconds) for a cluster job (default: 4h).",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=100_000,
        help="Memory (in MB) to use for the HTCondor job (default: 100 GB).",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=96,
        help="Number of CPUs to use for the HTCondor job (default: 96).",
    )
    parser.add_argument(
        "--n-live-points",
        type=int,
        default=4_000,
        help="Number of livepoints (default 4,000).",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=None,
        help="Parameters to use for the retrieval (default: all).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1_000,
        choices=[400, 1_000],
        help=r"Resolution R = ∆λ/λ to use for simulations (default: 1,000).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory where to save the results.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["dynesty", "multinest", "nautilus"],
        default="nautilus",
        help="Nested sampling sampler to use (default: nautilus).",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help="Whether to start a new submission.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=15,
        help="Time limit (in seconds) for simulations (default: 10s).",
    )
    args = parser.parse_args()

    return args


def sync_mpi_processes(comm: Any) -> None:
    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":

    print("\nRUN NESTED SAMPLING RETRIEVAL\n", flush=True)

    # Load and augment command line arguments
    args = get_cli_arguments()
    if args.parameters is None:
        args.parameters = list(NAMES)

    # Make sure we do not run nested sampling on the login node
    check_if_on_login_node(args.start_submission)

    # Collect arguments for submission file
    if args.sampler == "multinest":
        executable = "/usr/mpi/current/bin/mpiexec"
        job_arguments = [
            f"-n {args.n_cpus}",
            "--bind-to core:overload-allowed",
            "--mca coll ^hcoll",
            "--mca pml ob1",
            "--mca btl self,vader,tcp",
            sys.executable,
            Path(__file__).resolve().as_posix(),
        ]
    else:
        executable = sys.executable
        job_arguments = [Path(__file__).resolve().as_posix()]

    # -------------------------------------------------------------------------
    # Either prepare first submission...
    # -------------------------------------------------------------------------

    if args.start_submission:

        # Prepare the results directory
        results_dir = Path(__file__).parent / "results" / args.sampler
        results_dir.mkdir(exist_ok=True, parents=True)

        # Create a new directory for this run
        n_runs = len(list(results_dir.glob("*")))
        run_dir = results_dir / str(f"{n_runs:d}-" + generate_slug(2))
        run_dir.mkdir(exist_ok=True)
        job_arguments.append(f"--run-dir {run_dir}")

        # Dump all arguments to a file
        with open(run_dir / "arguments.json", "w") as json_file:
            json.dump(args.__dict__, json_file, indent=4)

        # Document the git status
        document_git_status(target_dir=run_dir, verbose=True)

        # Collect all arguments for the submission file
        for key, value in args.__dict__.items():
            if key not in ["start_submission", "run_dir"]:
                key = key.replace("_", "-")
                value = " ".join(value) if isinstance(value, list) else value
                job_arguments.append(f"--{key} {value}")

        print("Creating submission file...", end=" ", flush=True)
        condor_settings = CondorSettings(
            executable=executable,
            num_cpus=args.n_cpus,
            memory_cpus=args.memory,
            arguments=job_arguments,
            retry_on_exit_code=42,
            log_file_name=f"log.$$([NumJobStarts])",
        )
        file_path = create_submission_file(
            condor_settings=condor_settings,
            experiment_dir=run_dir,
        )
        print("Done!", flush=True)

        print("Submitting job...", end=" ", flush=True)
        condor_submit_bid(file_path=file_path, bid=args.bid)
        print("Done!\n", flush=True)

        sys.exit(0)

    # -------------------------------------------------------------------------
    # ...or actually run the nested sampling algorithm
    # -------------------------------------------------------------------------

    if args.run_dir is None:
        raise RuntimeError("Must specify --run-dir or --start-submission!")

    # In case of MultiNest + MPI, this will be overwritten
    comm = None
    rank = 0

    # Treat warnings as errors
    warnings.filterwarnings("error")
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed(args.random_seed)

    # Handle MPI communication for MultiNest
    if args.sampler == "multinest":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f"MPI rank: {rank}", flush=True)
        sync_mpi_processes(comm)

    print("Simulating ground truth spectrum...", end=" ", flush=True)
    theta_obs, x_obs = get_target_parameters_and_spectrum(
        resolution=args.resolution,
        time_limit=args.time_limit,
    )
    print("Done!", flush=True)
    sync_mpi_processes(comm)

    print("Creating simulator...", end=" ", flush=True)
    simulator = Simulator(
        noisy=False,
        R=args.resolution,
        time_limit=args.time_limit,
    )
    print("Done!", flush=True)
    sync_mpi_processes(comm)

    # Define prior and likelihood
    print("Setting up prior and likelihood...", end=" ", flush=True)

    # Get the lower and upper bounds for the selected parameters
    idx = np.array([list(NAMES).index(name) for name in args.parameters])
    lower = np.array(LOWER)[idx]
    upper = np.array(UPPER)[idx]

    # Define the prior transform function
    def prior(u: np.ndarray) -> np.ndarray:
        return np.array(lower + (upper - lower) * u)

    # Define the log-likelihood function
    # The noise level sigma is computed as `1.25754e-17 * 1e16` to match
    # the choice from `fm4ar.datasets.vasist_2023.simulation.Simulator`.
    # The value was original chosen to give a SNR of 10 (see paper).
    def likelihood(theta: np.ndarray, sigma: float = 0.125754) -> float:
        # Update theta_obs with the new values
        # This allows to run a retrieval for a subset of the parameters,
        # while keeping the others fixed to their ground truth values.
        combined_theta = theta_obs.copy()
        combined_theta[idx] = theta

        # If anything goes wrong, return an approximation for "-inf"
        # (MultiNest can't seem to handle proper -inf values and will complain)
        try:
            result = simulator(combined_theta)
        except Exception as e:
            print(f"{e.__class__.__name__}: {str(e)}", file=sys.stderr)
            return -1e300

        # If the simulation timed out, return "-inf"
        if result is None:
            return -1e300

        # If there are NaNs, return "-inf"
        wavelengths, x = result
        if np.isnan(x).any():
            return -1e300

        # Otherwise, return the log-likelihood
        # Note: The scaling (`sigma`) does matter here even if it is the
        # same for each wavelength bin! It must match the noise level used
        # when training an ML model to make the results comparable.
        return float(-0.5 * np.sum(((x - x_obs) / sigma) ** 2))

    print("Done!", flush=True)
    sync_mpi_processes(comm)

    print("Creating sampler...", end=" ", flush=True)
    sampler = get_sampler(args.sampler)(
        run_dir=Path(args.run_dir),
        prior=prior,
        likelihood=likelihood,
        n_dim=len(args.parameters),
        n_livepoints=args.n_live_points,
        parameters=list(args.parameters),
        random_seed=args.random_seed,
    )
    print("Done!\n", flush=True)
    sync_mpi_processes(comm)

    print("Running sampler:", flush=True)
    sampler.run(max_runtime=args.max_runtime, verbose=True)
    sampler.cleanup()
    sync_mpi_processes(comm)

    # If the sampler finished, we can save the results
    exit_code = 0
    if rank == 0:

        # If we are not done, exit with code 42 to signal that we need to
        # hold the job on exit and resubmit it
        if not sampler.complete:
            exit_code = 42

        else:
            print("Sampling complete!", flush=True)

            print("Saving results...", end=" ", flush=True)
            sampler.save_results()
            print("Done!", flush=True)

            print("Creating plot...", end=" ", flush=True)
            create_posterior_plot(
                points=sampler.points,
                weights=sampler.weights,
                parameters=args.parameters,
                file_path=Path(args.run_dir) / "posterior.pdf",
                ground_truth=theta_obs,
            )
            print("Done!", flush=True)

            print("\nAll done!\n", flush=True)

    # Make sure all processes are done before exiting
    sync_mpi_processes(comm)
    sys.exit(exit_code)