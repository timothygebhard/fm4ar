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
    print("\nMERGE HDF FILES\n", flush=True)

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
    print("Finding HDF files...", end=" ", flush=True)
    target_dir = get_datasets_dir() / "toy-dataset"
    file_list = sorted(target_dir.glob(f"{args.which}__*.hdf"))
    print("Done!", flush=True)
    print(f"Found {len(file_list)} files.\n", flush=True)

    # Initialize lists in which to collect data
    wavelengths = None
    list_of_thetas = []
    list_of_spectra = []
    list_of_noise = []
    list_of_posterior_samples = []

    print("Collecting HDF files:", flush=True)
    for file_path in tqdm(file_list, ncols=80):
        with h5py.File(file_path, "r") as hdf_file:

            # Load data from HDF file
            thetas = np.array(hdf_file["theta"]).astype(float)
            spectra = np.array(hdf_file["spectra"]).astype(float)

            # For test only: load noise and posterior samples
            if args.which == "test":
                noise = np.array(hdf_file["noise"]).astype(float)
                posterior_samples = (
                    np.array(hdf_file["posterior_samples"]).astype(float)
                )
            else:
                noise = np.empty(1)
                posterior_samples = np.empty(1)

            # Store data
            list_of_thetas.append(thetas)
            list_of_spectra.append(spectra)
            list_of_noise.append(noise)
            list_of_posterior_samples.append(posterior_samples)
            wavelengths = np.array(hdf_file["wavelengths"])

    # Convert to numpy arrays
    thetas = np.concatenate(list_of_thetas)
    del list_of_thetas
    spectra = np.concatenate(list_of_spectra)
    del list_of_spectra
    noise = np.concatenate(list_of_noise)
    del list_of_noise
    posterior_samples = np.concatenate(list_of_posterior_samples)
    del list_of_posterior_samples

    # Save merged HDF file
    print("\nMerging HDF files...", end=" ", flush=True)
    file_path = target_dir / f"{args.which}.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        hdf_file.create_dataset(name="theta", data=thetas, dtype=float)
        hdf_file.create_dataset(name="spectra", data=spectra, dtype=float)
        hdf_file.create_dataset(name="wavelengths", data=wavelengths)
        if args.which == "test":
            hdf_file.create_dataset(name="noise", data=noise, dtype=float)
            hdf_file.create_dataset(
                name="posterior_samples", data=posterior_samples, dtype=float
            )
    print("Done!\n", flush=True)

    print(f"Total mumber of spectra: {len(spectra):,}")

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
