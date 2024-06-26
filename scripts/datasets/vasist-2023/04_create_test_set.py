"""
Create different types of test sets for the `vasist_2023` dataset.
"""

import argparse
import datetime
import sys
import time
from importlib.metadata import version

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.datasets.vasist_2023.prior import THETA_0, Prior
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.environment import get_packages
from fm4ar.utils.git_utils import get_git_hash
from fm4ar.utils.paths import get_datasets_dir

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TEST SET\n", flush=True)

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theta-mode",
        type=str,
        default="default",
        choices=["default", "gaussian", "contracted", "benchmark"],
        help=(
            "Mode for sampling theta. There are four options available:"
            "  1. 'default': Sample directly from prior."
            "  2. 'gaussian': Sample from a Gaussian distribution centered "
            "       at the location of the benchmark spectrum."
            "  3. 'contracted': Sample from a contracted version of the "
            "       prior, i.e., avoid the outermost 5% of the prior range. "
            "  4. 'benchmark': Always use the same theta (but add different"
            "       noise realizations)."
            "Default: 'default'."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=400,
        choices=[400, 1_000],
        help="Spectral resolution (R = λ/∆λ). Default: 400.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.05,
        help="Minimum value for the std of the noise. Default: 0.05.",
    )
    parser.add_argument(
        "--max-sigma",
        type=float,
        default=0.50,
        help="Maximum value for the std of the noise. Default: 0.50.",
    )
    parser.add_argument(
        "--n-spectra",
        type=int,
        default=1000,
        help="Number of spectra to simulate. Default: 1000.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=10,
        help="Time limit for the simulation in seconds. Default: 10.",
    )
    args = parser.parse_args()

    # Set up prior
    # Note: The random seed here does not matter, because we do not call the
    # `sample()` method of the prior, but instead sample from a unit cube and
    # call `transform()` to get the atmospheric parameters.
    print("Setting up prior...", end=" ", flush=True)
    prior = Prior(random_seed=0)
    print("Done!")

    # Set up simulator
    print("Setting up simulator...", end=" ", flush=True)
    simulator = Simulator(R=args.resolution, time_limit=args.time_limit)
    print("Done!\n")

    # Create RNG for sampling both the parameters and the noise
    rng = np.random.default_rng(seed=args.random_seed)

    # Keep track of the results
    wlen = np.empty(0)
    thetas: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    sigmas: list[float] = []

    # Simulate spectra until the desired number of spectra is reached
    print("Generating test set:", flush=True)
    with tqdm(total=args.n_spectra, ncols=80) as progressbar:
        while len(thetas) < args.n_spectra:

            # Sample parameters from the prior (in the unit cube)
            if args.theta_mode == "default":
                u = rng.uniform(0, 1, size=len(prior.names))
            elif args.theta_mode == "gaussian":
                u = prior.distribution.cdf(THETA_0)
                u = u + rng.normal(0, 0.05, size=len(prior.names))
                u = np.clip(u, 0, 1)
            elif args.theta_mode == "contracted":
                u = rng.uniform(0.05, 0.95, size=len(prior.names))
            elif args.theta_mode == "benchmark":
                u = prior.distribution.cdf(THETA_0)
            else:
                raise ValueError(f"Invalid theta mode: {args.theta_mode}")

            # Transform the random numbers to the parameter space
            theta = prior.transform(u).astype(np.float32)

            # Simulate target spectrum
            result = simulator(theta)
            if result is None:
                print("Failed!", flush=True)
                continue
            else:
                wlen, flux = result

            # Convert to float32
            wlen = wlen.astype(np.float32)
            flux = flux.astype(np.float32)

            # Add noise to the spectrum
            sigma = rng.uniform(args.min_sigma, args.max_sigma)
            noise = sigma * rng.standard_normal(size=flux.shape)
            noise = noise.astype(np.float32)

            # Add noise to the spectrum
            noisy_flux = flux + noise

            # Store the results
            thetas.append(theta)
            fluxes.append(noisy_flux)
            noises.append(noise)
            sigmas.append(sigma)

            # Update progress bar
            progressbar.update(1)

    # Construct error bars from the noise level for each spectrum
    error_bars = np.array(
        [np.full_like(wlen, sigma, dtype=np.float32) for sigma in sigmas]
    )

    # Gather meta-information to save to HDF file (as attributes)
    n_bins = len(wlen)
    prt_version = version("petitRADTRANS")
    metadata = vars(args) | {
        "HEAD of fm4ar": get_git_hash(),
        "Timestamp (UTC)": datetime.datetime.utcnow().isoformat(),
        "Python version": sys.version,
        "petitRADTRANS version": version("petitRADTRANS"),
        "packages": "\n".join(get_packages()),
    }

    # Prepare the output directory
    output_dir = get_datasets_dir() / "vasist-2023" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save target data to HDF file
    print("\nSaving results...", end=" ", flush=True)
    prefix = f"test-{args.theta_mode}__R-{args.resolution}"
    file_path = output_dir / f"{prefix}__seed-{args.random_seed}.hdf"
    with h5py.File(file_path, "w") as f:
        f.attrs.update(metadata)
        f.create_dataset("wlen", data=wlen.reshape(1, -1))
        f.create_dataset("flux", data=np.array(fluxes), dtype=np.float32)
        f.create_dataset("error_bars", data=error_bars, dtype=np.float32)
        f.create_dataset("noise", data=np.array(noises), dtype=np.float32)
        f.create_dataset("theta", data=np.array(thetas), dtype=np.float32)
        f.create_dataset("sigma", data=np.array(sigmas), dtype=np.float32)
    print("Done!\n")

    print("Results saved to:\n", file_path, flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
