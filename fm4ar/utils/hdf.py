"""
Utility functions for working with HDF5 files.
"""

from pathlib import Path

import h5py
import numpy as np


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


def merge_hdf_files(
    target_dir: Path,
    name_pattern: str,
    output_file: Path,
    keys: list[str] | None = None,
    delete_after_merge: bool = False,
) -> None:
    """
    Merge the HDF files in the given directory and save the results.

    To save memory, the arrays are not loaded into memory at once, but
    rather concatenated directly from the HDF files.

    Args:
        target_dir: Path to the directory containing the HDF files.
        name_pattern: Pattern for the file names (e.g., "seed-*.hdf").
        output_file: Path to the output HDF file.
        keys: Keys of the arrays to merge. If None, merge all keys.
        delete_after_merge: Whether to delete the source HDF files
            after merging. Default: False.
    """

    # Collect the HDF files
    file_paths = sorted(target_dir.glob(name_pattern))

    # Open the first HDF file to get the keys, shapes, and dtypes
    keys_shapes_dtypes = {}
    with h5py.File(file_paths[0], "r") as f:
        for key in f.keys():
            if keys is None or key in keys:
                keys_shapes_dtypes[key] = (f[key].shape, f[key].dtype)

    # Prepare output HDF file
    with h5py.File(output_file, "w") as f:
        for key, (shape, dtype) in keys_shapes_dtypes.items():
            f.create_dataset(
                name=key,
                shape=(0, *shape[1:]),
                maxshape=(None, *shape[1:]),
                dtype=dtype,
            )

    # Loop over all HDF files and collect / merge the data
    for file_path in file_paths:
        with h5py.File(file_path, "r") as src:
            with h5py.File(output_file, "a") as dst:
                for key in dst.keys():
                    value = np.array(src[key], dtype=src[key].dtype)
                    dst[key].resize(dst[key].shape[0] + value.shape[0], axis=0)
                    dst[key][-value.shape[0]:] = value

    # Delete the source HDF file
    if delete_after_merge:
        for file_path in file_paths:
            file_path.unlink()
