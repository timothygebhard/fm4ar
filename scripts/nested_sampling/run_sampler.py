"""
Unified script to run different nested sampling algorithms.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, NAMES, LABELS
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.nested_sampling.config import load_config
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
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory where to save the results.",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help="Whether to start a new submission.",
    )
    args = parser.parse_args()

    return args


def sync_mpi_processes(comm: Any) -> None:
    if comm is not None:
        comm.Barrier()


def create_posterior_plot(
    points: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    ground_truth: np.ndarray,
    file_path: Path,
) -> None:
    """
    Create a corner plot of the posterior.
    """

    # Create the corner plot using ChainConsumer
    c = ChainConsumer()
    c.add_chain(
        chain=points,
        weights=weights,
        parameters=names,
        name="posterior",
    )
    c.configure(sigmas=[0, 1, 2, 3], summary=False)
    _ = c.plotter.plot(truth=ground_truth.tolist())

    # Save the plot
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":

    print("\nRUN NESTED SAMPLING RETRIEVAL\n", flush=True)

    # Load command line arguments and configuration file
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Make sure we do not run nested sampling on the login node
    check_if_on_login_node(args.start_submission)

    # Collect arguments for submission file
    if config.sampler.which == "multinest":
        executable = "/usr/mpi/current/bin/mpiexec"
        job_arguments = [
            f"-n {config.htcondor.n_cpus}",
            "--bind-to core:overload-allowed",
            "--mca coll ^hcoll",
            "--mca pml ob1",
            "--mca btl self,vader,tcp",
            sys.executable,
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
        ]
    else:
        executable = sys.executable
        job_arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
        ]

    # -------------------------------------------------------------------------
    # Either prepare first submission...
    # -------------------------------------------------------------------------

    if args.start_submission:

        print("Creating submission file...", end=" ", flush=True)
        condor_settings = CondorSettings(
            executable=executable,
            num_cpus=config.htcondor.n_cpus,
            memory_cpus=config.htcondor.memory,
            arguments=job_arguments,
            retry_on_exit_code=42,
            log_file_name="log.$$([NumJobStarts])",
        )
        file_path = create_submission_file(
            condor_settings=condor_settings,
            experiment_dir=args.experiment_dir,
        )
        print("Done!", flush=True)

        print("Submitting job...", end=" ", flush=True)
        condor_submit_bid(file_path=file_path, bid=config.htcondor.bid)
        print("Done!\n", flush=True)

        sys.exit(0)

    # -------------------------------------------------------------------------
    # ...or actually run the nested sampling algorithm
    # -------------------------------------------------------------------------

    # Document the git status
    document_git_status(target_dir=args.experiment_dir, verbose=True)

    # In case of MultiNest + MPI, this will be overwritten
    comm = None
    rank = 0

    # Treat warnings as errors
    warnings.filterwarnings("error")
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed(config.sampler.random_seed)

    # Handle MPI communication for MultiNest
    if config.sampler.which == "multinest":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f"MPI rank: {rank}", flush=True)
        sync_mpi_processes(comm)

    print("Creating simulator...", end=" ", flush=True)
    simulator = Simulator(
        noisy=False,
        R=config.simulator.resolution,
        time_limit=config.simulator.time_limit,
    )
    print("Done!", flush=True)
    sync_mpi_processes(comm)

    print("Simulating ground truth spectrum...", end=" ", flush=True)
    theta_obs = np.array([config.parameters[n].true_value for n in NAMES])
    if (result := simulator(theta_obs)) is None:
        raise RuntimeError("Failed to simulate ground truth!")
    _, x_obs = result
    print("Done!", flush=True)
    sync_mpi_processes(comm)

    # Define prior and likelihood
    print("Setting up prior and likelihood...", end=" ", flush=True)

    # Create binary masks for the different parameters actions
    infer_mask = np.array(
        [config.parameters[name].action == "infer" for name in NAMES]
    )
    marginalize_mask = np.array(
        [config.parameters[name].action == "marginalize" for name in NAMES]
    )

    # Get a prior sample for the marginalized parameters
    def sample_marginalized_parameters() -> np.ndarray:
        return np.array(
            np.random.uniform(
                np.array(LOWER)[marginalize_mask],
                np.array(UPPER)[marginalize_mask],
            )
        )

    # Get the lower and upper bounds for the parameters which we want to infer
    # and for which we need to transform the prior
    lower = np.array(LOWER)[infer_mask]
    upper = np.array(UPPER)[infer_mask]

    # Define the prior transform function
    def prior(u: np.ndarray) -> np.ndarray:
        return np.array(lower + (upper - lower) * u)

    # Define the log-likelihood function
    # The noise level sigma is computed as `1.25754e-17 * 1e16` to match
    # the choice from `fm4ar.datasets.vasist_2023.simulation.Simulator`.
    # The value was original chosen to give a SNR of 10 (see paper).
    def likelihood(theta: np.ndarray, sigma: float = 0.125754) -> float:

        # Construct theta for the simulation.
        # First, we copy the ground truth values for all parameters.
        combined_theta = theta_obs.copy()

        # Then, we overwrite the values for the parameters over which we
        # want to marginalize with a random sample from the prior
        if marginalize_mask.any():
            combined_theta[marginalize_mask] = sample_marginalized_parameters()

        # Finally, we overwrite the values for the parameters which we want
        # to infer with the values from the current sample that is controlled
        # by the nested sampling algorithm
        combined_theta[infer_mask] = theta

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
        _, x = result
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
    sampler = get_sampler(config.sampler.which)(
        run_dir=args.experiment_dir,
        prior=prior,
        likelihood=likelihood,
        n_dim=sum(infer_mask),
        n_livepoints=config.sampler.n_livepoints,
        inferred_parameters=np.array(NAMES)[infer_mask].tolist(),
        random_seed=config.sampler.random_seed,
    )
    print("Done!\n", flush=True)
    sync_mpi_processes(comm)

    print("Running sampler:", flush=True)
    sampler.run(
        max_runtime=config.htcondor.max_runtime,
        verbose=True,
        run_kwargs=config.sampler.run_kwargs,
    )
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
                points=np.array(sampler.points),
                weights=np.array(sampler.weights),
                names=np.array(LABELS)[infer_mask].tolist(),
                file_path=args.experiment_dir / "posterior.pdf",
                ground_truth=theta_obs[infer_mask],
            )
            print("Done!", flush=True)

            print("\nAll done!\n", flush=True)

    # Make sure all processes are done before exiting
    sync_mpi_processes(comm)
    sys.exit(exit_code)
