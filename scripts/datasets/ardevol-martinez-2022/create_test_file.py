"""
Merge all the test files into a single HDF file (for convenience).
"""

import time
from itertools import product

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE TEST FILES FOR ARDEVOL MARTINEZ (2022) DATASET\n")

    dataset_dir = get_datasets_dir() / "ardevol-martinez-2022"

    # Ensure output file is empty so that we can append to it
    print("Preparing output file...", end=" ")
    output_file_path = dataset_dir / "test" / "merged.hdf"
    with h5py.File(output_file_path, "w") as hdf_file:
        pass
    print("Done!")

    # Load metadata (for wavelengths and noise levels)
    print("Loading metadata...", end=" ")
    metadata = pd.read_pickle(dataset_dir / "train" / "metadata.p")
    print("Done!\n")

    # Loop over all combinations of instrument and chemistry model
    for instrument, chemistry_model in product(["NIRSPEC", "WFC3"], [1, 2]):

        target_dir = (
            dataset_dir
            / "test"
            / instrument
            / f"type{chemistry_model}"
            / "obs_w_noise"
        )

        # Collect the flux values and corresponding noise realizations
        list_of_flux = []
        list_of_noise = []
        print(f"Collecting {instrument} type-{chemistry_model}:")
        for i in tqdm(range(1000), ncols=80):
            data = np.loadtxt(target_dir / str(i))
            flux = 100 * data[:, 1]
            noise = 1e6 * data[:, 2]
            list_of_flux.append(flux)
            list_of_noise.append(noise)

        # Merge all the fluxes and noises into a single array
        flux = np.array(list_of_flux)
        noise = np.array(list_of_noise)

        # Get ground truth parameter values
        theta = np.array(np.loadtxt(target_dir / "parameters"))

        # Save the data to the output file
        print("Saving to output file...", end=" ")
        with h5py.File(output_file_path, "a") as hdf_file:

            # Store wavelengths and noise levels
            if instrument not in hdf_file.keys():
                group = hdf_file.create_group(instrument)
                group.create_dataset(
                    name="wlen",
                    data=metadata["Wavelength"][instrument],
                    dtype=float,
                )
                group.create_dataset(
                    name="noise_levels",
                    data=metadata["Noise"][instrument],
                    dtype=float,
                )

            # Store parameters, flux values, and noises
            group = hdf_file.create_group(f"{instrument}/{chemistry_model}")
            group.create_dataset(name="theta", data=theta, dtype=float)
            group.create_dataset(name="flux", data=flux, dtype=float)
            group.create_dataset(name="noise", data=noise, dtype=float)

            # Add names and ranges of parameters
            group.attrs["names"] = metadata["names"][chemistry_model]
            group.attrs["ranges"] = metadata["ranges"][chemistry_model]

        print("Done!\n")

    print(f"This took {time.time() - script_start:.1f} seconds!\n")
