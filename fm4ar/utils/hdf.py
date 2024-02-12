"""
Utility functions for working with HDF5 files.
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def save_to_hdf(
    file_path: Path,
    **kwargs: np.ndarray,
) -> None:
    """
    Save the given arrays to an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.
        kwargs: Arrays to save.
    """

    with h5py.File(file_path, "w") as f:
        for key, value in kwargs.items():
            f.create_dataset(name=key, data=value, dtype=value.dtype)


def load_from_hdf(
    file_path: Path,
    keys: list[str],
    idx: int | slice | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Load the given arrays from an HDF5 file.

    Args:
        file_path: Path to the HDF file.
        keys: Keys of the arrays to load.
        idx: Indices of the arrays to load.

    Returns:
        data: Loaded arrays.
    """

    data = {}
    with h5py.File(file_path, "r") as f:
        for key in keys:
            if idx is None:
                data[key] = np.array(f[key], dtype=f[key].dtype)
            else:
                data[key] = np.array(f[key][idx], dtype=f[key].dtype)

    return data


def load_merged_hdf_files(
    target_dir: Path,
    name_pattern: str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    """
    Merge the HDF files in the given directory and return the results.

    Args:
        target_dir: Path to the directory containing the HDF files.
        name_pattern: Pattern for the file names (e.g., "seed-*.hdf").
        keys: Keys of the arrays to merge.

    Returns:
        merged: Merged arrays.
    """

    # Collect the HDF files
    file_paths = sorted(target_dir.glob(name_pattern))

    # Loop over all HDF files and collect / mergge the data
    merged: dict[str, np.ndarray] = {}
    for file_path in tqdm(file_paths, ncols=80):
        with h5py.File(file_path, "r") as f:
            for key in keys:
                value = np.array(f[key], dtype=f[key].dtype)
                if key not in merged:
                    merged[key] = value
                else:
                    merged[key] = np.concatenate([merged[key], value])

    return merged
