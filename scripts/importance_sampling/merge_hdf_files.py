"""
Merge the HDF files from the importance sampling runs into a single
HDF file (and recompute the weights accordindly).
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE IMPORTANCE SAMPLING HDF FILES\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    args = parser.parse_args()

    # Collect HDF files that need to be merged
    results_dir = args.experiment_dir / "importance_sampling"
    file_paths = sorted(results_dir.glob("random_seed-*.hdf"))

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

    # Re-compute the weights
    print("Re-computing weights...", end=" ", flush=True)
    raw_weights = merged["raw_weights"]
    weights = raw_weights * len(raw_weights) / np.sum(raw_weights)
    merged["weights"] = weights
    print("Done!\n")

    # Compute the effective sample size and sample efficiency
    n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
    sample_efficiency = float(n_eff / len(weights))
    print(f"Effective sample size: {n_eff:.2f}")
    print(f"Sample efficiency:     {100 * sample_efficiency:.2f}%\n")

    # Save the merged data to an HDF file
    print("Saving to an HDF file...", end=" ", flush=True)
    file_path = results_dir / "importance_sampling_results.hdf"
    with h5py.File(file_path, "w") as f:
        for key in merged.keys():
            f.create_dataset(
                name=key,
                data=merged[key],
                dtype=merged[key].dtype,
            )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
