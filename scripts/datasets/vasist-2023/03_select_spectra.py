"""
Select spectra from the Vasist-2023 dataset that meet certain criteria.
"""

import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nSELECT SPECTRA\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file-name",
        type=str,
        default="selected.hdf",
        help="Name of the output HDF file with the selected spectra.",
    )
    parser.add_argument(
        "--which",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Which dataset to select spectra from (train or test).",
    )
    args = parser.parse_args()

    # Hard-code some constants about the data
    N_PARAMETERS = 16
    N_BINS = 947

    # Define target directory
    target_dir = get_datasets_dir() / "vasist-2023" / args.which

    # Open the HDF file with all the spectra
    with h5py.File(target_dir / args.output_file_name, "w") as dst:

        # Prepare datasets in the output HDF file
        dst.create_dataset(
            name="theta",
            shape=(0, N_PARAMETERS),
            maxshape=(None, N_PARAMETERS),
            dtype=np.float32,
        )
        dst.create_dataset(
            name="spectra",
            shape=(0, N_BINS),
            maxshape=(None, N_BINS),
            dtype=np.float32,
        )

        # Open the HDF file with all the spectra
        with h5py.File(target_dir / "merged.hdf", "r") as src:

            # # Copy over wavelengths
            dst.create_dataset(
                name="wavelengths",
                data=src["wavelengths"][...],
                dtype=np.float32,
            )

            # Select spectra: For this, we loop over chunks of 4096 spectra at
            # a time (to limit memory consumption) and copy over the ones that
            # meet the criteria
            print("Selecting spectra:")
            idx = np.arange(0, len(src["spectra"]), 4096)
            for a, b in tqdm(
                list(zip(idx[:-1], idx[1:], strict=True)), ncols=80
            ):

                # Define criteria for spectra to keep
                mean = np.mean(src["spectra"][a:b], axis=1)
                mask = (1e-5 <= mean) & (mean <= 1e5)

                # Select spectra that meet the criteria
                spectra = np.array(src["spectra"][a:b])[mask]
                theta = np.array(src["theta"][a:b])[mask]
                n = len(spectra)

                # Resize the datasets in the output HDF file
                dst["spectra"].resize((dst["spectra"].shape[0] + n), axis=0)
                dst["theta"].resize((dst["theta"].shape[0] + n), axis=0)

                # Save the selected spectra and theta
                dst["spectra"][-n:] = spectra
                dst["theta"][-n:] = theta

            # Print some information about how many spectra we selected
            print("Before:", src["spectra"].shape)
            print("After: ", dst["spectra"].shape)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
