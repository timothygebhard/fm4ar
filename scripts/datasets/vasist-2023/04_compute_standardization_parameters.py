"""
Compute the standardization parameters from the training dataset.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


def get_standardization_parameters(
    file_path: Path,
    key: str,
    buffer_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the standardization parameters for the dataset without
    loading the full dataset into memory.

    Args:
        file_path: Path to the HDF file containing the train dataset.
        key: Key of the dataset in the HDF file.
        buffer_size: Size of the buffer to use when computing the
            mean and std.

    Returns:
        A tuple, `(mean, std)`, containing the standardization
        parameters.
    """

    # Keep track of some statistics
    s0: float = 0.0
    s1: np.ndarray | float = 0.0
    s2: np.ndarray | float = 0.0

    # Loop over the dataset in a "chunked" fashion and compute the statistics.
    # Motivation: Looping over each sample individually is very slow.
    with h5py.File(file_path.as_posix(), "r") as hdf_file:

        # Prepare the indices for the chunks
        n = len(hdf_file[key])
        idx = (
            np.r_[0 : n : buffer_size, n] if n > buffer_size
            else np.array([0, n])
        )
        limits = list(zip(idx[:-1], idx[1:], strict=True))

        # Note: We loop twice because the single loop version seems to be
        # numerically unstable (unless we cast everything to float128)

        # Loop 1: Compute the mean
        for a, b in tqdm(limits, ncols=80, desc="mean"):
            x = np.array(hdf_file[key][a:b]).astype(np.float64)
            s0 += len(x)
            s1 += np.sum(x, axis=0)
        mean = np.array(s1 / s0).astype(np.float64)

        # Loop 2: Compute the std
        for a, b in tqdm(limits, ncols=80, desc="std "):
            x = np.array(hdf_file[key][a:b]).astype(np.float64)
            x = x - mean
            s2 += np.sum(x ** 2, axis=0)
        std = np.sqrt(s2 / (s0 - 1)).astype(np.float32)
        mean = np.array(s1 / s0).astype(np.float32)

    return mean, std


if __name__ == "__main__":

    script_start = time.time()
    print("\nCOMPUTE STANDARDIZATION PARAMETERS\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4096,
        help="Size of the buffer to use when computing the mean and std.",
    )
    parser.add_argument(
        "--input-file-name",
        type=str,
        default="merged.hdf",
        help=(
            "Name of the input HDF file with the spectra and parameters. "
            "This can be different we run a pre-selection of the spectra."
        ),
    )
    parser.add_argument(
        "--which",
        type=str,
        default="train",
        help=(
            "Name of the directory that contains the input HDF file."
            "Usually, this is either 'train' or 'test'."
        ),
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    dataset_dir = get_datasets_dir() / "vasist-2023"
    input_dir = dataset_dir / args.which
    precomputed_dir = dataset_dir / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    # Compute the mean and std of the flux
    print("Computing standardization parameters for the flux:")
    flux_mean, flux_std = get_standardization_parameters(
        file_path=input_dir / args.input_file_name,
        key="flux",
        buffer_size=args.buffer_size,
    )
    print()

    # Save the standardization parameters for theta
    print("Computing standardization parameters for theta:")
    theta_mean, theta_std = get_standardization_parameters(
        file_path=input_dir / args.input_file_name,
        key="theta",
    )
    print()

    # Create a new HDF file for the standardization parameters
    print("Saving standardization parameters...", end=" ")
    suffix = args.input_file_name.split('.')[0]
    file_name = f"standardization_parameters__{suffix}.hdf"
    file_path = precomputed_dir / file_name
    with h5py.File(file_path, "w") as f:
        f.create_dataset(name="flux/mean", data=flux_mean, dtype=np.float32)
        f.create_dataset(name="flux/std", data=flux_std, dtype=np.float32)
        f.create_dataset(name="theta/mean", data=theta_mean, dtype=np.float32)
        f.create_dataset(name="theta/std", data=theta_std, dtype=np.float32)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
