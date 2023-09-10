"""
Create HDF files for the Goyal-2020 dataset.
"""

import time

import h5py
import numpy as np

from fm4ar.utils.paths import (
    get_datasets_dir,
    get_path_from_environment_variable,
)


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE HDF FILES FOR GOYAL-2020 DATASET\n")

    # Get the path to the Goyal-2020 dataset
    original_dir = (
        get_path_from_environment_variable("ML4PTP_DATASETS_DIR")
        / "goyal-2020"
        / "output"
    )
    output_dir = get_datasets_dir() / "goyal-2020" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the HDF files
    for file_name in ("train.hdf", "test.hdf"):

        print(f"Creating file: {file_name}")

        # Read in the original file
        print("Reading original file...", end=" ")
        with (h5py.File(original_dir / file_name, "r") as f):

            # Wavelengths (everything up to 8 micron, downsampled by factor 4)
            # TODO: What is a good choice here?
            wlen = np.array(f["emission_spectra"]["emission_wavelength"])[0]
            mask = (wlen <= 8.0) * (np.arange(len(wlen)) % 4 == 0)
            wlen = wlen[mask]

            # Spectra: We use the "contrast emission spectra"
            # TODO: Does this make sense?
            planetary_flux = np.array(f["emission_spectra"]["planetary_flux"])
            stellar_flux = np.array(f["emission_spectra"]["stellar_flux"])
            flux = planetary_flux / (1 + stellar_flux)
            flux = flux[:, mask]

            # Define "theta" (target parameters)
            theta = np.column_stack(
                [
                    np.array(f["emission_spectra"]["planet_radius"]) / 1e7,
                    np.mean(
                        np.array(f["pt_profiles"]["temperature"]),
                        axis=1,
                    ),
                    np.log10(
                        np.max(
                            np.array(f["pt_profiles"]["pressure"]),
                            axis=1,
                        )
                    ),
                    np.log10(
                        np.mean(
                            np.array(f["chemical_abundances"]["H2O"]),
                            axis=1,
                        )
                    ),
                    np.log10(
                        np.mean(
                            np.array(f["chemical_abundances"]["CO2"]),
                            axis=1,
                        ),
                    ),
                    np.log10(
                        np.mean(
                            np.array(f["chemical_abundances"]["CH4"]),
                            axis=1,
                        )
                    ),
                ]
            )

            # Define names and ranges of the parameters
            names = [
                r"R_P",  # Planet radius in km
                r"T_{mean}",  # Mean temperature in K
                r"\log(P_0)",  # Surface log-pressure in log(bar)
                r"\log(X_{H_2O})",  # Mean water abundance
                r"\log(X_{CO_2})",  # Mean carbon dioxide abundance
                r"\log(X_{CH_4})",  # Mean methane abundance
            ]
            ranges = [
                (np.min(theta[:, i]), np.max(theta[:, i]))
                for i in range(theta.shape[1])
            ]

        print("Done!")

        # Write the HDF file
        print("Writing new file...", end=" ")
        with h5py.File(output_dir / file_name, "w") as f:
            f.attrs["names"] = names
            f.attrs["ranges"] = ranges
            f.create_dataset("wlen", data=wlen, dtype=np.float32)
            f.create_dataset("flux", data=flux, dtype=np.float32)
            f.create_dataset("theta", data=theta, dtype=np.float32)
        print("Done!\n")

    print(f"This took {time.time() - script_start:.1f} seconds.\n")
