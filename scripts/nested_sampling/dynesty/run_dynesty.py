"""
Test the nautilus package.
"""

import json
import multiprocessing
import os
import time
import warnings
from pathlib import Path
from textwrap import indent

import dill
import dynesty.utils
import numpy as np
from coolname import generate_slug
from dynesty import DynamicNestedSampler

from fm4ar.datasets.vasist_2023.prior import NAMES
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.nested_sampling import (
    get_cli_arguments,
    get_subsets,
    get_target_parameters_and_spectrum,
    create_posterior_plot,
)
from fm4ar.utils.multiproc import get_number_of_available_cores


if __name__ == "__main__":

    script_start = time.time()
    print("\nRUN DYNESTY RETRIEVAL\n", flush=True)

    # Treat warnings as errors
    warnings.filterwarnings("error")

    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = "1"

    # Set pickle module to dill
    dynesty.utils.pickle_module = dill

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

    # Update parameters if necessary
    if args.parameters is None:
        args.parameters = NAMES

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

    # Create a simulator
    print("Creating simulator...", end=" ", flush=True)
    simulator = Simulator(
        noisy=args.add_noise,
        R=args.resolution,
        time_limit=args.time_limit,
    )
    print("Done!", flush=True)

    # Define prior and likelihood
    print("Setting up prior and likelihood...", end=" ", flush=True)

    # Get the lower and upper bounds for the selected parameters
    idx, lower, upper, _ = get_subsets(args.parameters)

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

    # Run the sampler
    # Note: Somehow, the default "spawn" context always crashes with an
    # "AttributeError: Can't get attribute 'prior' on <module '__mp_main__'>"
    checkpoint_path = run_dir / "checkpoint.save"
    n_processes = get_number_of_available_cores()
    with multiprocessing.get_context("fork").Pool(n_processes) as pool:
        sampler = DynamicNestedSampler(
            loglikelihood=likelihood,
            prior_transform=prior,
            ndim=len(args.parameters),
            nlive=args.n_live_points,
            pool=pool,
            queue_size=get_number_of_available_cores(),
        )
        sampler.run_nested(checkpoint_file=checkpoint_path.as_posix())
    print()

    # Save posterior
    print("Saving posterior...", end=" ", flush=True)
    file_path = run_dir / "posterior.pickle"
    with open(file_path, 'wb') as handle:
        dill.dump(obj=sampler.results, file=handle)
    print("Done!", flush=True)

    # Plot results
    print("Creating plot...", end=" ", flush=True)
    file_path = run_dir / "posterior.pdf"
    create_posterior_plot(
        points=sampler.results["samples"],
        weights=np.exp(sampler.results["logwt"]),
        parameters=args.parameters,
        file_path=file_path,
        ground_truth=theta_obs,
    )
    print("Done!", flush=True)

    # Print and save total runtime
    results = {
        "it": int(sampler.it),
        "ncall": int(sampler.ncall),
        "eff": float(sampler.eff),
        "total_runtime": time.time() - script_start,
    }
    with open(run_dir / "results.txt", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nThis took {results['total_runtime']:.2f} seconds!\n", flush=True)
