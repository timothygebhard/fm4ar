"""
Script to run different nested sampling implementations on HTCondor.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from fm4ar.likelihoods import get_likelihood_distribution
from fm4ar.nested_sampling.config import load_config
from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.nested_sampling.utils import (
    create_posterior_plot,
    get_parameter_masks,
)
from fm4ar.priors import get_prior
from fm4ar.simulators import get_simulator
from fm4ar.utils.environment import document_environment
from fm4ar.utils.git_utils import document_git_status
from fm4ar.utils.htcondor import (
    check_if_on_login_node,
    condor_submit_bid,
    create_submission_file,
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
        if comm.Get_rank() == 0:
            print("Synchronizing MPI processes...", end=" ", flush=True)
        comm.Barrier()
        if comm.Get_rank() == 0:
            print("Done!", flush=True)


if __name__ == "__main__":

    # Load command line arguments and configuration file
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Make sure we do not run nested sampling on the login node
    check_if_on_login_node(args.start_submission)

    # -------------------------------------------------------------------------
    # Either prepare first submission...
    # -------------------------------------------------------------------------

    if args.start_submission:

        print("\nPREPARE NESTED SAMPLING RETRIEVAL\n", flush=True)

        # Document the git status and the Python environment
        document_git_status(target_dir=args.experiment_dir, verbose=True)
        document_environment(target_dir=args.experiment_dir)

        # Collect arguments for submission file
        # MultiNest and UltraNest both parallelize the sampling process using
        # MPI, which is why we need a bunch of additional arguments for them.
        # Also, it looks like at least UltraNest benefits from setting the
        # number of processes to about half the number of CPUs --- otherwise,
        # the load average is significantly higher than the number of CPUs and
        # things actually slow down (about 10% in some preliminary tests).
        if config.sampler.library in ("multinest", "ultranest"):
            factor = 2 if config.sampler.library == "ultranest" else 1
            executable = "/usr/mpi/current/bin/mpiexec"
            job_arguments = [
                f"-np {config.htcondor.n_cpus // factor}",
                "--bind-to core:overload-allowed",
                "--mca coll ^hcoll",
                "--mca pml ob1",
                "--mca btl self,vader,tcp",
                "--verbose",
                sys.executable,
                Path(__file__).resolve().as_posix(),
                f"--experiment-dir {args.experiment_dir}",
            ]
        else:
            executable = sys.executable
            job_arguments = [
                Path(__file__).resolve().as_posix(),
                f"--experiment-dir {args.experiment_dir.resolve()}",
            ]

        print("Creating submission file...", end=" ", flush=True)

        # Augment the HTCondor configuration
        htcondor_config = config.htcondor
        htcondor_config.executable = executable
        htcondor_config.arguments = job_arguments
        htcondor_config.retry_on_exit_code = 42
        htcondor_config.log_file_name = "log.$$([NumJobStarts])"
        htcondor_config.extra_kwargs = (
            {}
            if config.sampler.library not in ("multinest", "ultranest")
            else {"transfer_executable": "False"}
        )

        # Create submission file
        file_path = create_submission_file(
            htcondor_config=htcondor_config,
            experiment_dir=args.experiment_dir.resolve(),
        )

        print("Done!\n", flush=True)

        logs_dir = args.experiment_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        condor_submit_bid(file_path=file_path, bid=config.htcondor.bid)

        sys.exit(0)

    # -------------------------------------------------------------------------
    # ...or actually run the nested sampling algorithm
    # -------------------------------------------------------------------------

    # Limit number of threads to 1 to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"

    # In case of MultiNest + MPI, this will be overwritten
    comm = None
    rank = 0

    # Set random seed for reproducibility
    np.random.seed(config.sampler.random_seed + rank)

    # Define a simple overloaded print function that flushes the output and
    # limits the output to the root process (rank 0) in case of MPI
    def log(*args: Any, **kwargs: Any) -> None:
        if rank == 0:
            print(*args, **kwargs, flush=True)

    # Handle MPI communication for MultiNest and UltraNest
    if config.sampler.library in ("multinest", "ultranest"):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sync_mpi_processes(comm)

    log("\nRUN NESTED SAMPLING RETRIEVAL\n")

    log("Creating prior distribution...", end=" ")
    config.prior.random_seed += rank
    prior = get_prior(config=config.prior)
    log("Done!")

    log("Creating simulator...", end=" ")
    simulator = get_simulator(config=config.simulator)
    log("Done!")

    log("Simulating ground truth...", end=" ")
    theta_obs = np.array([config.ground_truth[n] for n in prior.names])
    if (result := simulator(theta_obs)) is None:
        log("Failed!")
        raise RuntimeError(f"[{rank:2d}] Failed to simulate ground truth!")
    _, flux_obs = result
    log("Done!")

    log("Creating likelihood distribution...", end=" ")
    likelihood_distribution = get_likelihood_distribution(
        flux_obs=flux_obs,
        config=config.likelihood,
    )
    log("Done!")

    log("Creating log-likelihood function...", end=" ")

    # Create masks that indicate which parameters are being inferred, which are
    # being marginalized over, and which are being conditioned on (= fixed)
    (
        infer_mask,
        marginalize_mask,
        condition_mask,
        condition_values,
    ) = get_parameter_masks(prior=prior, config=config.prior)

    # Define the log-likelihood function
    # This combines the construction of the theta vector for the simulation,
    # the simulation itself, and the evaluation of the likelihood function.
    def log_likelihood(infer_values: np.ndarray) -> float:

        # Construct full theta for the simulation
        # We start with the values that we are conditioning on, that is, the
        # values that will simply be fixed to a constant value
        theta = condition_values.copy()

        # Then, we overwrite the values for the parameters over which we
        # want to marginalize with a random sample from the prior
        if marginalize_mask.any():
            theta[marginalize_mask] = prior.sample()[marginalize_mask]

        # Finally, we overwrite the values for the parameters which we want
        # to infer with the values from the current sample that is controlled
        # by the nested sampling algorithm. The size of this
        theta[infer_mask] = infer_values

        # If anything goes wrong, return an approximation for "-inf"
        # (MultiNest can't seem to handle proper -inf values and will complain)
        # Note: We return different numbers for the three cases so that in case
        # they ever show up in the output, we can distinguish them.
        try:
            result = simulator(theta)
        except Exception as e:
            print(
                f"\n\n{e.__class__.__name__}: {str(e)}\n",
                file=sys.stderr,
                flush=True,
            )
            return -1e299

        # If the simulation timed out, return "-inf"
        if result is None:
            print("\n\nSimulation timed out!\n", file=sys.stderr)
            return -1e298

        # If there are NaNs, return "-inf"
        _, x = result
        if np.isnan(x).any():
            print("\n\nSimulation result contains NaNs!\n", file=sys.stderr)
            return -1e297

        # Otherwise, return the log-likelihood
        return float(likelihood_distribution.logpdf(x))

    log("Done!")

    log("Instantiating sampler...", end=" ")
    sampler = get_sampler(config.sampler.library)(
        run_dir=args.experiment_dir,
        prior_transform=partial(prior.transform, mask=infer_mask),
        log_likelihood=log_likelihood,
        n_dim=sum(infer_mask),
        n_livepoints=config.sampler.n_livepoints,
        inferred_parameters=np.array(prior.names)[infer_mask].tolist(),
        sampler_kwargs=config.sampler.sampler_kwargs,
        random_seed=config.sampler.random_seed,
    )
    log("Done!")

    # Synchronize all processes before running the sampler
    sync_mpi_processes(comm)

    # Run the sampler until the maximum runtime is reached
    log("\n\nRunning sampler:\n")
    sampler.run(
        max_runtime=config.sampler.max_runtime,
        verbose=True,
        run_kwargs=config.sampler.run_kwargs,
    )
    sampler.cleanup()

    # Note: It seems that adding any more `sync_mpi_processes(comm)` calls
    # after this point will cause the MultiNest sampler to hang indefinitely.
    # This also applies to operations like `comm.allgather()` that could be
    # used to synchronize the `complete` flag across all processes.

    # Determine the exit code: 42 means "hold and restart the job"
    exit_code = 0 if sampler.complete else 42

    # If we are done, save the results and create a plot
    # For the case of MultiNest, we only do this on the "root" process
    if sampler.complete and rank == 0:

        log("\n\nSampling complete!")
        log("Saving results...", end=" ")
        sampler.save_results()
        log("Done!")

        log("Creating plot...", end=" ")
        create_posterior_plot(
            samples=np.array(sampler.samples),
            weights=np.array(sampler.weights),
            names=np.array(prior.labels)[infer_mask],
            extents=(
                np.array(prior.distribution.support()[0][infer_mask]),
                np.array(prior.distribution.support()[1][infer_mask]),
            ),
            file_path=args.experiment_dir / "posterior.pdf",
            ground_truth=theta_obs[infer_mask],
        )
        log("Done!")

        log("\nAll done!\n\n\n")

    # Make sure all processes are done before exiting
    print(f"Exiting job {rank} with code {exit_code}!", flush=True)
    sys.exit(exit_code)
