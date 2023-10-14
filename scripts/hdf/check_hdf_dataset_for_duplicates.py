"""
Check if a given HDF dataset contains duplicate entries.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="theta",
        help="Name of the dataset to check for duplicates.",
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        help="Path to the HDF file.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nCHECK HDF DATASET FOR DUPLICATES\n")

    args = get_cli_arguments()

    # Open the HDF file
    print("Reading in the HDF file...", end=" ", flush=True)
    with h5py.File(args.file_path, "r") as f:
        dataset = np.array(f[args.dataset])
    print("Done!", flush=True)

    # Check for duplicates
    print("Checking for duplicates...", end=" ", flush=True)
    unique, counts = np.unique(dataset, return_counts=True)
    duplicates = unique[counts > 1]
    print("Done!")

    # Print results
    print(f"Found {len(duplicates)} duplicates.\n")
    if len(duplicates) > 0:
        print("Duplicates:")
        print(duplicates)

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
