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
    progress_bar: bool = True,
    buffer_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the standardization parameters for the dataset without
    loading the full dataset into memory.

    Args:
        file_path: Path to the HDF file containing the train dataset.
        key: Key of the dataset in the HDF file.
        progress_bar: Whether to show a progress bar.
        buffer_size: Size of the buffer to use when computing the
            mean and std.

    Returns:
        A tuple, `(mean, std)`, containing the standardization
        parameters.
    """

    # Keep track of some statistics to compute the mean and std without
    # loading the full dataset into memory;
    # Source: https://stackoverflow.com/a/5543790/4100721
    s0: int = 0
    s1: np.ndarray | float = 0.0
    s2: np.ndarray | float = 0.0

    # Loop over the dataset in a "chunked" fashion and compute the statistics.
    # Motivation: Looping over each sample individually is very slow.
    with h5py.File(file_path.as_posix(), "r") as hdf_file:

        n = len(hdf_file[key])
        idx = (
            np.arange(0, n, buffer_size) if n > buffer_size
            else np.array([0, n])
        )

        if progress_bar:
            limits = tqdm(list(zip(idx[:-1], idx[1:], strict=True)))
        else:
            limits = list(zip(idx[:-1], idx[1:], strict=True))

        for a, b in limits:
            x = np.array(hdf_file[key][a:b])
            s0 += len(x)
            s1 += np.sum(x, axis=0)
            s2 += np.sum(x ** 2, axis=0)

    # Compute the mean and std
    mean = np.array(s1 / s0)
    std = np.sqrt((s0 * s2 - s1 * s1) / (s0 * (s0 - 1)))

    # Compute the standardization parameters
    return mean, std


if __name__ == "__main__":

    script_start = time.time()
    print("\nCOMPUTE STANDARDIZATION PARAMETERS\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file-name",
        type=str,
        default="merged.hdf",
        help=(
            "Name of the input HDF file with the spectra and parameters. "
            "This can be different we run a pre-selection of the spectra."
        ),
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    dataset_dir = get_datasets_dir() / "vasist-2023"
    train_dir = dataset_dir / "train"
    precomputed_dir = dataset_dir / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    # Compute the mean and std of the flux
    print("Computing standardization parameters for the flux:")
    flux_mean, flux_std = get_standardization_parameters(
        file_path=train_dir / args.input_file_name,
        key="spectra",
    )

    # Save the standardization parameters for theta
    print("Computing standardization parameters for theta:")
    theta_mean, theta_std = get_standardization_parameters(
        file_path=train_dir / args.input_file_name,
        key="theta",
    )

    # Create a new HDF file for the standardization parameters
    print("Saving standardization parameters...", end=" ")
    suffix = args.input_file_name.split('.')[0]
    file_name = f"standardization_parameters__{suffix}.hdf"
    file_path = precomputed_dir / file_name
    with h5py.File(file_path, "w") as f:
        f.create_dataset(name=f"flux/mean", data=flux_mean, dtype=float)
        f.create_dataset(name=f"flux/std", data=flux_mean, dtype=float)
        f.create_dataset(name=f"theta/mean", data=theta_mean, dtype=float)
        f.create_dataset(name=f"theta/std", data=theta_mean, dtype=float)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
