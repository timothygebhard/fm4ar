"""
Create a test set for the `vasist_2023` dataset.
"""

import argparse
import time

import numpy as np
from tqdm import tqdm

from fm4ar.utils.hdf import save_to_hdf
from fm4ar.datasets.vasist_2023.prior import Prior, THETA_0
from fm4ar.datasets.vasist_2023.simulator import Simulator
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
        choices=["default", "gaussian", "contracted"],
        help=(
            "Mode for sampling theta. There are three options available:"
            "  1. 'default': Sample directly from prior. "
            "  2. 'gaussian': Sample from a Gaussian distribution centered "
            "       at the location of the benchmark spectrum."
            "  3. 'contracted': Sample from a contracted version of the "
            "       prior, i.e., avoid the outermost 5% of the prior range. "
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
        default=100,
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
    print("Setting up prior...", end=" ", flush=True)
    prior = Prior(random_seed=args.random_seed)
    print("Done!")

    # Set up simulator
    print("Setting up simulator...", end=" ", flush=True)
    simulator = Simulator(R=args.resolution, time_limit=args.time_limit)
    print("Done!\n")

    # Create RNG for sampling sigma
    rng = np.random.default_rng(seed=args.random_seed + 1)

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

    # Prepare the output directory
    output_dir = get_datasets_dir() / "vasist-2023" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save target data to HDF file
    print("\nSaving results...", end=" ", flush=True)
    prefix = f"test-{args.theta_mode}__R-{args.resolution}"
    file_path = output_dir / f"{prefix}__seed-{args.random_seed}.hdf"
    save_to_hdf(
        file_path=file_path,
        wlen=wlen.reshape(1, -1).astype(np.float32),
        flux=np.array(fluxes).astype(np.float32),
        noise=np.array(noises).astype(np.float32),
        theta=np.array(thetas).astype(np.float32),
        sigma=np.array(sigmas).astype(np.float32),
    )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
