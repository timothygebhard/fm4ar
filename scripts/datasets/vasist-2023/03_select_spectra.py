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

    train_dir = get_datasets_dir() / "vasist-2023" / args.which

    # Open the HDF file with all the spectra
    with h5py.File(train_dir / args.output_file_name, "w") as dst:
        with h5py.File(train_dir / "merged.hdf", "r") as src:

            # Copy over wavelengths; prepare datasets for spectra and theta
            dst.create_dataset(
                name="wavelengths",
                data=src["wavelengths"][...],
                dtype=np.float32,
            )
            spectra = []
            thetas = []

            # Select spectra that meet the criteria
            print("Selecting spectra:")
            idx = np.arange(0, len(src["spectra"]), 4096)
            for a, b in tqdm(list(zip(idx[:-1], idx[1:])), ncols=80):

                # Define criteria for spectra to keep
                mean = np.mean(src["spectra"][a:b], axis=1)
                mask = (1e-5 <= mean) & (mean <= 1e5)

                spectra.append(np.array(src["spectra"][a:b])[mask])
                thetas.append(np.array(src["theta"][a:b])[mask])

            # Save the selected spectra and thetas
            print("\nSaving results...", end=" ")
            dst.create_dataset(
                name="spectra",
                data=np.concatenate(spectra, axis=0),
                dtype=float,
            )
            dst.create_dataset(
                name="theta",
                data=np.concatenate(thetas, axis=0),
                dtype=float,
            )
            print("Done!\n")

            print("Before:", src["spectra"].shape)
            print("After: ", dst["spectra"].shape)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
