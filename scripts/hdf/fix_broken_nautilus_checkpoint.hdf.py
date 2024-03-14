"""
This script attempts to "fix" a broken nautilus checkpoint file where
there is a mismatch between the number of points and the number of
weights. This seems like a rare issue, but re-running from scratch is
expensive, so trying to recover should be worth it.
"""

from argparse import ArgumentParser
from shutil import copyfile
from pathlib import Path
from time import time

import h5py


if __name__ == "__main__":

    script_start = time()
    print("\nFIX BROKEN NAUTILUS CHECKPOINT\n")

    # Get the path to the broken file
    parser = ArgumentParser()
    parser.add_argument(
        "--file-path",
        type=Path,
        required=True,
        help="Path to the broken checkpoint file.",
    )
    args = parser.parse_args()

    # Check if the file exists
    if not args.file_path.exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")
    else:
        print(f"Attempting to fix the following file: {args.file_path}")

    # Create a backup copy of the broken file
    print("Creating backup of broken checkpoint...", end=" ", flush=True)
    copyfile(
        src=args.file_path,
        dst=args.file_path.with_suffix(".hdf5.backup"),
    )
    print("Done!\n", flush=True)

    # Open the broken file
    with h5py.File(args.file_path, "a") as f:

        # For all datasets where the number of points and the number of log_l
        # do not match, we replace the dataset with an empty one. This will
        # cause the sampler to re-sample the corresponding shells, which is
        # cheaper than re-running from scratch.
        for key in filter(lambda k: "points" in k, f["/sampler"].keys()):

            points_key = f"/sampler/{key}"
            weights_key = f"/sampler/{key.replace('points', 'log_l')}"
            n_points = f[points_key].shape[0]
            n_weights = f[weights_key].shape[0]

            if n_points != n_weights:

                print(f"Replacing {points_key} with empty dataset...", end="")
                del f[points_key]
                f.create_dataset(
                    name=points_key,
                    shape=(0, 16),
                    maxshape=(None, 16),
                )
                print(" Done!", flush=True)

                print(f"Replacing {weights_key} with empty dataset...", end="")
                del f[weights_key]
                f.create_dataset(
                    name=weights_key,
                    shape=(0, ),
                    maxshape=(None, ),
                )
                print(" Done!", flush=True)

    print(f"\nThis took {time() - script_start:.1f} seconds.\n")
