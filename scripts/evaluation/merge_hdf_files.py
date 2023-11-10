"""
Merge partial evaluation HDF files into one.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    start_time = time.time()

    print("\nMERGE EVALUATION HDF FILES\n")

    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="test",
        help="Which dataset results to merge.",
    )
    args = parser.parse_args()

    # Find all partial HDF files
    evaluation_dir = args.experiment_dir / "evaluation"
    file_paths = sorted(evaluation_dir.glob(f"results_{args.which}*.hdf"))

    # Initialze the dict in which we will store all the data
    data: dict[str, list[np.ndarray]] = {}
    with h5py.File(file_paths[0], "r") as f:
        for key in f.keys():
            data[key] = []

    # Loop over all HDF files and collect the data
    print("Collecting data from HDF files:", flush=True)
    for file_path in tqdm(file_paths, ncols=80):
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                data[key].append(np.array(f[key], dtype=f[key].dtype))
    print()

    # Concatenate the data
    print("Concatenating data...", end=" ", flush=True)
    merged: dict[str, np.ndarray] = {}
    for key in data.keys():
        merged[key] = np.concatenate(data[key])
    print("Done!")

    # Save the merged data to an HDF file
    print("Saving to an HDF file...", end=" ", flush=True)
    file_path = evaluation_dir / f"results_on_{args.which}_set.hdf"
    with h5py.File(file_path, "w") as f:
        for key in merged.keys():
            f.create_dataset(
                name=key,
                data=merged[key],
                dtype=merged[key].dtype,
            )
    print("Done!")

    print(f"\nThis took {time.time() - start_time:.2f} seconds!\n")
