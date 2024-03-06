"""
Unified script to run different nested sampling algorithms.
"""

import argparse
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from fm4ar.nested_sampling.config import load_config
from fm4ar.nested_sampling.likelihood import get_likelihood_distribution
from fm4ar.nested_sampling.priors import get_prior
from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.nested_sampling.simulators import get_simulator
from fm4ar.nested_sampling.utils import (
    create_posterior_plot,
    get_parameter_masks,
)
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


if __name__ == "__main__":

    print("\nRUN NESTED SAMPLING RETRIEVAL\n", flush=True)

    # Load command line arguments and configuration file
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Make sure we do not run nested sampling on the login node
    check_if_on_login_node(args.start_submission)

    # Collect arguments for submission file
    if config.sampler.library == "multinest":
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
            f"--experiment-dir {args.experiment_dir.resolve()}",
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
            extra_kwargs=(
                {}
                if config.sampler.library != "multinest"
                else {"transfer_executable": "False"}
            ),
        )
        file_path = create_submission_file(
            condor_settings=condor_settings,
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
    if config.sampler.library == "multinest":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f"MPI rank: {rank}", flush=True)
        sync_mpi_processes(comm)

    print("Creating prior distribution...", end=" ", flush=True)
    prior = get_prior(config=config.prior)
    print("Done!", flush=True)

    print("Creating simulator...", end=" ", flush=True)
    simulator = get_simulator(config=config.simulator)
    print("Done!", flush=True)

    sync_mpi_processes(comm)

    # TODO: Maybe this could be extended to work also with actual observations,
    #   that is, load the spectrum from a file and use it?
    print("Simulating ground truth spectrum...", end=" ", flush=True)
    theta_obs = np.array([config.ground_truth[n] for n in prior.names])
    if (result := simulator(theta_obs)) is None:
        raise RuntimeError("Failed to simulate ground truth!")
    _, x_obs = result
    print("Done!", flush=True)

    sync_mpi_processes(comm)

    print("Creating likelihood distribution...", end=" ", flush=True)
    likelihood_distribution = get_likelihood_distribution(
        x_obs=x_obs,
        config=config.likelihood,
    )
    print("Done!", flush=True)

    print("Preparing log_likelihood function...", end=" ", flush=True)

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
        try:
            result = simulator(theta)
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
        return float(likelihood_distribution.logpdf(x))

    print("Done!", flush=True)

    sync_mpi_processes(comm)

    print("Creating sampler...", end=" ", flush=True)
    sampler = get_sampler(config.sampler.library)(
        run_dir=args.experiment_dir,
        prior_transform=partial(prior.transform, mask=infer_mask),
        log_likelihood=log_likelihood,
        n_dim=sum(infer_mask),
        n_livepoints=config.sampler.n_livepoints,
        inferred_parameters=np.array(prior.names)[infer_mask].tolist(),
        random_seed=config.sampler.random_seed,
        **config.sampler.sampler_kwargs,
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
                names=np.array(prior.labels)[infer_mask],
                file_path=args.experiment_dir / "posterior.pdf",
                ground_truth=theta_obs[infer_mask],
            )
            print("Done!", flush=True)

            print("\nAll done!\n", flush=True)

    # Make sure all processes are done before exiting
    sync_mpi_processes(comm)
    sys.exit(exit_code)
