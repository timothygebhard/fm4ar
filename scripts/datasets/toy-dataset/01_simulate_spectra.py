"""
Create an HDF file with some toy data.
"""

import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.datasets.toy_dataset import (
    simulate_toy_spectrum,
    get_posterior_samples,
)
from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TOY DATA\n")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-spectra",
        type=int,
        default=100,
        help="Number of 'spectra' to create.",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=4,
        help="Number of parameters to use.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for the random seed (for parallel data generation).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Number of 'wavelength' points.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Standard deviation for noise to be added.",
    )
    parser.add_argument(
        "--which",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which dataset to create ('train' or 'test').",
    )
    args = parser.parse_args()

    # Set random seed
    rng = np.random.RandomState(args.random_seed + args.offset)

    # Collect spectra and parameters
    list_of_spectra = []
    list_of_theta = []
    list_of_posterior_samples: list[np.ndarray | None] = []
    list_of_noise: list[np.ndarray | None] = []
    wavelengths = np.full(args.resolution, np.nan)

    # Create spectra and parameters
    print("Generating pseudo-spectra:")
    for _ in tqdm(list(range(args.n_spectra)), ncols=80):

        # Generate a random spectrum
        theta = rng.normal(0, 1, args.n_parameters)
        wavelengths, spectrum = simulate_toy_spectrum(theta, args.resolution)

        # Compute posterior (only for test set)
        if args.which == "test":
            noise = rng.normal(0, args.sigma, args.resolution)
            noisy_spectrum = spectrum + noise
            posterior_samples: np.ndarray | None = get_posterior_samples(
                true_spectrum=noisy_spectrum,
                true_theta=theta,
                sigma=args.sigma,
                n_livepoints=1000,
                n_samples=1000,
            )

        else:
            noise = None
            posterior_samples = None

        # Store everything
        list_of_theta.append(theta)
        list_of_spectra.append(spectrum)
        list_of_noise.append(noise)
        list_of_posterior_samples.append(posterior_samples)

    print()

    # Convert to numpy arrays
    spectra = np.array(list_of_spectra)
    theta = np.array(list_of_theta)
    noise = np.array(list_of_noise)
    posterior_samples = np.array(list_of_posterior_samples)

    # Ensure the output directory exists
    output_dir = get_datasets_dir() / "toy-dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create HDF file
    print("Creating HDF file...", end=" ")
    effective_random_seed = args.random_seed + args.offset
    file_name = output_dir / f"{args.which}__{effective_random_seed:04d}.hdf"
    with h5py.File(file_name, "w") as hdf_file:
        hdf_file.create_dataset(
            name="wavelengths",
            data=wavelengths,
            dtype=float,
        )
        hdf_file.create_dataset(
            name="spectra",
            data=spectra,
            dtype=float,
        )
        hdf_file.create_dataset(
            name="theta",
            data=theta,
            dtype=float,
        )
        if args.which == "test":
            hdf_file.create_dataset(
                name="noise",
                data=noise,
                dtype=float,
            )
            hdf_file.create_dataset(
                name="posterior_samples",
                data=posterior_samples,
                dtype=float,
            )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
