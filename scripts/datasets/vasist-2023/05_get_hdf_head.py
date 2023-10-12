"""
This script can be used to export the "head" (i.e., the first N entries
of every dataset) of an HDF file to a new HDF file. This can be useful,
for example, for local debugging purposes.

Note: This file uses the same N for all datasets, and it ignores groups
and attributes entirely.
"""

import argparse
import time
from pathlib import Path

import h5py


if __name__ == "__main__":

    script_start = time.time()
    print("\nSELECT SUBSET OF HDF FILE\n", flush=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-file",
        type=Path,
        help="Path to the source HDF file.",
    )
    parser.add_argument(
        "--dst-file",
        type=Path,
        help="Path to the destination HDF file.",
    )
    parser.add_argument(
        "--n-samples",
        default=100,
        type=int,
        help="Number of samples to select.",
    )
    args = parser.parse_args()

    # Print arguments
    print(f"Source file:       {args.src_file}")
    print(f"Destination file:  {args.dst_file}")
    print(f"Number of samples: {args.n_samples}\n")

    # Copy over the first N samples
    print("Copying over the first N samples...")
    with (
        h5py.File(args.src_file, "r") as src,
        h5py.File(args.dst_file, "w") as dst,
    ):
        for key in src.keys():
            dst.create_dataset(key, data=src[key][: args.n_samples])
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
