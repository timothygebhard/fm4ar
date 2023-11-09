"""
Merge the HDF files for each random seed into a single HDF file.
"""

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
        "--target-dir",
        type=str,
        default="train",
        help="Name of the target directory (default: train).",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE HDF FILES\n", flush=True)

    args = get_cli_arguments()

    # Collect source HDF files
    print("Finding HDF files...", end=" ")
    target_dir = get_datasets_dir() / "vasist-2023" / args.target_dir
    file_list = sorted(target_dir.glob("random-seed_*.hdf"))
    print("Done!", flush=True)
    print(f"Found {len(file_list)} files.\n")

    # Prepare output HDF file
    print("Preparing output HDF file...", end=" ")
    dst_file_path = target_dir / "merged.hdf"
    with h5py.File(dst_file_path, "w") as f:
        f.create_dataset(
            name="noise",
            shape=(0, args.n_bins),
            maxshape=(None, args.n_bins),
            dtype=np.float32,
        )
        f.create_dataset(
            name="theta",
            shape=(0, args.n_parameters),
            maxshape=(None, args.n_parameters),
            dtype=np.float32,
        )
        f.create_dataset(
            name="flux",
            shape=(0, args.n_bins),
            maxshape=(None, args.n_bins),
            dtype=np.float32,
        )
    print("Done!\n", flush=True)

    # Keep track of the number of spectra we discard due to NaNs
    n_dropped = 0

    # Open output HDF file
    with h5py.File(dst_file_path, "a") as dst:

        # Loop over source HDF files and append data to output HDF file
        # This should be a lot more memory efficient than loading all data
        # into memory at once, and then writing it to the output HDF file.
        print("Collecting HDF files:", flush=True)
        for src_file_path in tqdm(file_list, ncols=80):
            with h5py.File(src_file_path, "r") as src:

                # Load data from source HDF file
                theta = np.array(src["theta"])
                flux = np.array(src["flux"])
                if "noise" in src.keys():
                    noise = np.array(src["noise"])

                # Only keep files with at least one spectrum
                if len(theta) == 0:
                    continue

                # Exclude spectra with NaNs
                mask = np.isnan(flux).any(axis=1)
                noise = noise[~mask]
                theta = theta[~mask]
                flux = flux[~mask]
                n = (~mask).sum()
                n_dropped += mask.sum()

                # Resize datasets in output HDF file
                dst["theta"].resize(dst["theta"].shape[0] + n, axis=0)
                dst["flux"].resize(dst["flux"].shape[0] + n, axis=0)
                if "noise" in src.keys():
                    dst["noise"].resize(dst["noise"].shape[0] + n, axis=0)

                # Write data to output HDF file
                dst["theta"][-n:] = theta
                dst["flux"][-n:] = flux
                if "noise" in src.keys():
                    dst["noise"][-n:] = noise

                # Copy over wavelengths
                if "wlen" not in dst:
                    dst.create_dataset(
                        name="wlen",
                        data=src["wlen"][...],
                        dtype=np.float32,
                    )

            # Drop noise dataset if its empty
            if "noise" in dst.keys() and len(dst["noise"]) == 0:
                del dst["noise"]

        # Get total number of spectra
        n_spectra = dst["theta"].shape[0]

    # Print some information about the data
    print()
    print(f"Total number of spectra: {n_spectra:,}", flush=True)
    print(f"Excluded due to NaNs:    {n_dropped:,}", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
