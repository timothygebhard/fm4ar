"""
Merge all the test files into a single HDF file (for convenience).
"""

import pickle
import time
from itertools import product

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE TEST FILES\n")

    # Prepare the output file (ensure it is empty)
    print("Preparing output file...", end=" ", flush=True)
    output_file_path = (
        get_datasets_dir()
        / "ardevol-martinez-2022"
        / "test"
        / "merged.hdf"
    )
    with h5py.File(output_file_path, "w") as hdf_file:
        pass
    print("Done!", flush=True)

    # Load metadata (for wavelengths and noise levels)
    print("Loading metadata...", end=" ", flush=True)
    metadata_file_path = (
        get_datasets_dir()
        / "ardevol-martinez-2022"
        / "training"
        / "metadata.p"
    )
    metadata = pickle.load(open(metadata_file_path, "rb"))
    print("Done!", flush=True)

    # Loop over all combinations of instrument and chemistry model
    for instrument, chemistry_model in product(["NIRSPEC", "WFC3"], [1, 2]):
        target_dir = (
            get_datasets_dir()
            / "ardevol-martinez-2022"
            / "test"
            / instrument
            / f"type{chemistry_model}"
            / "obs_w_noise"
        )

        # Get ground truth parameter values
        file_path = target_dir / "parameters"
        theta = np.loadtxt(file_path)

        # Get simulatated spectra plus noise
        list_of_spectra = []
        list_of_noises = []
        desc = f"Loading {instrument} type-{chemistry_model}"
        for i in tqdm(range(1000), ncols=80, desc=desc):
            file_path = target_dir / str(i)
            data = np.loadtxt(file_path)
            spectrum = 100 * data[:, 1]  # TODO: Is this correct?
            noise = 1e6 * data[:, 2]  # TODO: Is this correct?
            list_of_spectra.append(spectrum)
            list_of_noises.append(noise)
        print("")

        # Merge all the spectra and noises into a single array
        spectra = np.array(list_of_spectra)
        noises = np.array(list_of_noises)

        # Save the data to the output file
        print("Saving to output file...", end=" ", flush=True)
        with h5py.File(output_file_path, "a") as hdf_file:

            # Store wavelengths and noise levels
            if instrument not in hdf_file.keys():
                group = hdf_file.create_group(instrument)
                group.create_dataset(
                    name="wavelengths",
                    data=metadata["Wavelength"][instrument],
                )
                group.create_dataset(
                    name="noise_levels",
                    data=metadata["Noise"][instrument],
                )

            # Store parameters, spectra, and noises
            group = hdf_file.create_group(f"{instrument}/{chemistry_model}")
            group.create_dataset("theta", data=theta)
            group.create_dataset("spectra", data=spectra)
            group.create_dataset("noise", data=noises)

            # Add names and ranges of parameters
            group.attrs["names"] = metadata["names"][chemistry_model]
            group.attrs["ranges"] = metadata["ranges"][chemistry_model]

        print("Done!\n", flush=True)

    print(f"This took {time.time() - script_start:.1f} seconds!\n", flush=True)
