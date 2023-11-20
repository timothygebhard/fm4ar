"""
Minimize the results of a HDF file by dropping everything that we do
not need for plotting. (Reduces file size significantly.)
"""

import argparse
import time
from pathlib import Path

import h5py


if __name__ == "__main__":

    script_start = time.time()
    print("\nMINIMIZE HDF FILE\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    args = parser.parse_args()

    # Construct paths
    is_path = args.experiment_dir / "importance_sampling"
    src_path = is_path / "importance_sampling_results.hdf"
    dst_path = is_path / "importance_sampling_results_minimized.hdf"

    # Open the HDF file
    # We only need to copy over the `theta` and `weights` datasets
    print("Minimizing the HDF file...", end=" ", flush=True)
    with h5py.File(dst_path, "w") as dst:
        with h5py.File(src_path, "r") as src:
            dst.create_dataset(
                name="theta",
                data=src["theta"],
                compression="gzip",
                compression_opts=9,
            )
            dst.create_dataset(
                name="weights",
                data=src["weights"],
                compression="gzip",
                compression_opts=9,
            )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
