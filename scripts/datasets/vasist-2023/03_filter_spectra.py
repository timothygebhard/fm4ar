"""
Filter out spectra that do not meet certain criteria.
"""

# TODO: Ideally, we would define the filter criteria in the `data`
#   section of a config file. This would allow us to easily change
#   the criteria without having to change the code, plus it would
#   be more explicit.

import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-bins",
        type=int,
        default=947,
        help="Number of bins in the spectra (default: 947).",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=16,
        help="Number of simulation parameters (default: 16).",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default="filtered.hdf",
        help="Name of the output HDF file with the selected spectra.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="train",
        help="Directory that contains the merged.hdf file (e.g., 'train').",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nSELECT SPECTRA\n")

    args = get_cli_arguments()

    # Define target directory
    target_dir = get_datasets_dir() / "vasist-2023" / args.target_dir

    # Open the HDF file with all the spectra
    with h5py.File(target_dir / args.output_file_name, "w") as dst:

        # Prepare datasets in the output HDF file
        dst.create_dataset(
            name="noise",
            shape=(0, args.n_bins),
            maxshape=(None, args.n_bins),
            dtype=np.float32,
        )
        dst.create_dataset(
            name="theta",
            shape=(0, args.n_parameters),
            maxshape=(None, args.n_parameters),
            dtype=np.float32,
        )
        dst.create_dataset(
            name="flux",
            shape=(0, args.n_bins),
            maxshape=(None, args.n_bins),
            dtype=np.float32,
        )

        # Open the HDF file with all the spectra
        with h5py.File(target_dir / "merged.hdf", "r") as src:

            # Copy over wavelengths
            dst.create_dataset(
                name="wlen",
                data=src["wlen"][...],
                dtype=np.float32,
            )

            # Prepare indices for looping over the file in chunks
            buffer_size = 4096
            n = len(src["flux"])
            idx = (
                np.r_[0:n:buffer_size, n]
                if n > buffer_size
                else np.array([0, n])
            )
            limits = list(zip(idx[:-1], idx[1:], strict=True))

            print("Selecting spectra:")
            for a, b in tqdm(limits, ncols=80):

                # Define criteria for spectra to keep
                mean = np.mean(src["flux"][a:b], axis=1)
                # mask = (1e-5 <= mean) & (mean <= 1e5)
                mask = 1e-5 <= mean

                # Select spectra that meet the criteria
                flux = np.array(src["flux"][a:b])[mask]
                theta = np.array(src["theta"][a:b])[mask]
                if "noise" in src.keys():
                    noise = np.array(src["noise"][a:b])[mask]

                # Resize the datasets in the output HDF file
                n = len(flux)
                dst["flux"].resize((dst["flux"].shape[0] + n), axis=0)
                dst["theta"].resize((dst["theta"].shape[0] + n), axis=0)
                if "noise" in src.keys():
                    dst["noise"].resize((dst["noise"].shape[0] + n), axis=0)

                # Save the selected spectra and theta
                dst["flux"][-n:] = flux
                dst["theta"][-n:] = theta
                if "noise" in src.keys():
                    dst["noise"][-n:] = noise

            # Drop the noise dataset if it is empty
            if "noise" in dst.keys() and len(dst["noise"]) == 0:
                del dst["noise"]

            # Print some information about how many spectra we selected
            print("\nNumber of spectra:")
            before = src["theta"].shape[0]
            after = dst["theta"].shape[0]
            dropped = before - after
            print(f"Before:  {before:,}")
            print(f"After:   {after:,}")
            print(f"Dropped: {dropped:,} ({100 * dropped / before:.1f}%)")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
