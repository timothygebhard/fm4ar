"""
Merge the HDF files for each random seed into a single HDF file.
"""

import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE HDF FILES\n")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Which files to merge ('train' or 'test').",
    )
    args = parser.parse_args()

    # Collect list of HDF files to be merged
    print("Finding HDF files...", end=" ")
    target_dir = get_datasets_dir() / "toy-dataset" / args.which
    file_list = sorted(target_dir.glob(f"{args.which}__*.hdf"))
    print("Done!")
    print(f"Found {len(file_list)} files.\n")

    # Initialize lists in which to collect data
    wlen = None
    list_of_thetas = []
    list_of_flux = []
    list_of_noise = []
    list_of_samples = []

    print("Collecting HDF files:")
    for file_path in tqdm(file_list, ncols=80):
        with h5py.File(file_path, "r") as hdf_file:

            # Load data from HDF file
            thetas = np.array(hdf_file["theta"]).astype(float)
            flux = np.array(hdf_file["flux"]).astype(float)

            # For test only: load noise and posterior samples
            if args.which == "test":
                noise = np.array(hdf_file["noise"]).astype(float)
                samples = np.array(hdf_file["samples"]).astype(float)
            else:
                noise = np.empty(1)
                samples = np.empty(1)

            # Store data
            list_of_thetas.append(thetas)
            list_of_flux.append(flux)
            list_of_noise.append(noise)
            list_of_samples.append(samples)
            wlen = np.array(hdf_file["wlen"])

    print()

    # Convert to numpy arrays
    thetas = np.concatenate(list_of_thetas)
    del list_of_thetas
    flux = np.concatenate(list_of_flux)
    del list_of_flux
    noise = np.concatenate(list_of_noise)
    del list_of_noise
    samples = np.concatenate(list_of_samples)
    del list_of_samples

    # Save merged HDF file
    print("Merging HDF files...", end=" ")
    file_path = target_dir.parent / f"{args.which}.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        hdf_file.create_dataset(name="theta", data=thetas, dtype=float)
        hdf_file.create_dataset(name="flux", data=flux, dtype=float)
        hdf_file.create_dataset(name="wlen", data=wlen, dtype=float)
        if args.which == "test":
            hdf_file.create_dataset(name="noise", data=noise, dtype=float)
            hdf_file.create_dataset(name="samples", data=samples, dtype=float)
    print("Done!\n")
    print(f"Total number of spectra: {len(flux):,}")

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
