"""
Test the nautilus package.
"""

import json
import os
import time
import warnings
from pathlib import Path
from textwrap import indent

import numpy as np
from coolname import generate_slug
from nautilus import Sampler

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, NAMES
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.nested_sampling import (
    get_cli_arguments,
    get_target_parameters_and_spectrum,
    create_posterior_plot,
)
from fm4ar.utils.multiproc import get_number_of_available_cores


if __name__ == "__main__":

    script_start = time.time()
    print("\nRUN NAUTILUS RETRIEVAL\n", flush=True)

    # Treat warnings as errors
    warnings.filterwarnings("error")

    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = "1"

    # Get command line arguments and set random seed
    args = get_cli_arguments()
    np.random.seed(args.random_seed)

    # Get directory where to save the results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise ValueError(f"{run_dir} does not exist!")
    else:
        n_runs = len(list(results_dir.glob("*")))
        run_dir = results_dir / str(f"{n_runs:d}-" + generate_slug(2))
        run_dir.mkdir(exist_ok=True)

    # Print and store all relevant information about this run
    print(f"Run directory: {run_dir}", flush=True)
    print("Run parameters:\n", flush=True)
    print(indent(json.dumps(args.__dict__, indent=2), "  ") + "\n", flush=True)
    with open(run_dir / "arguments.txt", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Simulate ground truth spectrum
    print("Simulating ground truth spectrum...", end=" ", flush=True)
    theta_obs, x_obs = get_target_parameters_and_spectrum(
        resolution=args.resolution,
        time_limit=args.time_limit,
    )
    print("Done!", flush=True)

    # Create a simulator (with noise!)
    print("Creating simulator...", end=" ", flush=True)
    simulator = Simulator(
        noisy=True,
        R=args.resolution,
        time_limit=args.time_limit,
    )
    print("Done!", flush=True)

    # Define prior and likelihood
    print("Setting up prior and likelihood...", end=" ", flush=True)

    # Select the parameters to use (default: all)
    if args.parameters is None:
        args.parameters = NAMES

    # Get the lower and upper bounds for the selected parameters
    idx = np.array([list(NAMES).index(name) for name in args.parameters])
    lower = np.array(LOWER)[idx]
    upper = np.array(UPPER)[idx]

    # Define the prior transform function
    def prior(u: np.ndarray) -> np.ndarray:
        return np.array(lower + (upper - lower) * u)

    # Define the log-likelihood function
    def likelihood(theta: np.ndarray) -> float:
        # Get theta for simulator by updating theta_obs with the new values.
        # This allows to only run a retrieval for a subset of the parameters,
        # while keeping the others fixed to their ground truth values.
        combined_theta = theta_obs.copy()
        combined_theta[idx] = theta

        # If anything goes wrong, return -inf
        try:
            result = simulator(combined_theta)
        except Exception as e:
            print(e)
            return float("-inf")

        # If the simulation timed out, return -inf
        if result is None:
            return float("-inf")

        # If there are NaNs, return -inf
        wavelengths, x = result
        if np.isnan(x).any():
            return float("-inf")

        # Otherwise, return the negative log-likelihood
        return float(-0.5 * np.sum((x - x_obs) ** 2))

    print("Done!\n", flush=True)

    # Run sampler
    print("Running sampler:\n", flush=True)
    # noinspection PyTypeChecker
    sampler = Sampler(
        prior=prior,
        likelihood=likelihood,
        n_dim=len(args.parameters),
        n_live=args.n_live_points,
        pool=get_number_of_available_cores(),
        filepath=(run_dir / "checkpoint.hdf5").as_posix(),
        seed=args.random_seed,
    )
    sampler.run(verbose=True, f_live=0.1)
    print()

    # Save posterior
    print("Saving posterior...", end=" ", flush=True)
    points, log_w, log_l = sampler.posterior()
    file_path = run_dir / "posterior.npz"
    np.savez(file_path, points=points, log_w=log_w, log_l=log_l)
    print("Done!", flush=True)

    # Plot results
    print("Creating plot...", end=" ", flush=True)
    file_path = run_dir / "posterior.pdf"
    create_posterior_plot(
        points=points,
        weights=np.exp(log_w),
        parameters=args.parameters,
        file_path=file_path,
        ground_truth=theta_obs,
    )
    print("Done!", flush=True)

    # Print and save total runtime
    results = {
        "N_like": int(sampler.n_like),
        "N_eff": int(sampler.effective_sample_size()),
        "log Z": float(sampler.evidence()),
        "total_runtime": time.time() - script_start,
    }
    with open(run_dir / "results.txt", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nThis took {results['total_runtime']:.2f} seconds!\n", flush=True)

    # Manually shutdown the pool
    for pool in (sampler.pool_l, sampler.pool_s):
        if pool is not None:
            pool.close()
