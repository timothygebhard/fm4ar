"""
Create a test set for the `vasist_2023` dataset.
"""

import argparse
import time

import numpy as np

from fm4ar.utils.hdf import save_to_hdf
from fm4ar.datasets.vasist_2023.prior import Prior
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TEST SET\n", flush=True)

    # Get command line arguments
    parser = argparse.ArgumentParser()
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
    while len(thetas) < args.n_spectra:

        i = len(thetas) + 1
        print(f"[{i:3d}] Simulating spectrum...", end=" ", flush=True)

        # Sample parameters from the prior
        theta = prior.sample()

        # Simulate target spectrum
        result = simulator(theta)
        if result is None:
            print("Failed!", flush=True)
            continue
        else:
            wlen, flux = result

        # Add noise to the spectrum
        sigma = rng.uniform(args.min_sigma, args.max_sigma)
        noise = sigma * rng.standard_normal(size=flux.shape)

        # Add noise to the spectrum
        noisy_flux = flux + noise

        # Store the results
        thetas.append(theta)
        fluxes.append(noisy_flux)
        noises.append(noise)
        sigmas.append(sigma)

        print("Done!", flush=True)

    # Prepare the output directory
    output_dir = get_datasets_dir() / "vasist-2023" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save target data to HDF file
    print("\nSaving results...", end=" ", flush=True)
    file_name = f"default__R-{args.resolution}__seed-{args.random_seed}.hdf"
    file_path = output_dir / file_name
    save_to_hdf(
        file_path=file_path,
        wlen=wlen.reshape(1, -1),
        flux=np.array(fluxes),
        noise=np.array(noises),
        theta=np.array(thetas),
        sigma=np.array(sigmas),
    )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
