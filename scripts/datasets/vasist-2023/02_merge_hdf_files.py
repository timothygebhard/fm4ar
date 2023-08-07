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
        "--output-dir",
        type=str,
        default="training",
        help="Name of the output directory (default: training).",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE HDF FILES\n", flush=True)

    args = get_cli_arguments()

    print("Finding HDF files...", end=" ", flush=True)
    target_dir = get_datasets_dir() / "vasist-2023" / args.output_dir
    file_list = sorted(target_dir.glob("random-seed_*.hdf"))
    print("Done!", flush=True)
    print(f"Found {len(file_list)} files.\n", flush=True)

    num_nan_spectra = 0
    wavelengths = None
    list_of_thetas = []
    list_of_spectra = []
    print("Collecting HDF files:", flush=True)
    for file_path in tqdm(file_list, ncols=80):
        with h5py.File(file_path, "r") as hdf_file:

            # Load data from HDF file
            thetas = np.array(hdf_file["theta"]).astype(float)
            spectra = np.array(hdf_file["spectra"]).astype(float)

            # Only keep files with at least one spectrum
            if len(spectra) == 0:
                continue

            # Exclude spectra with NaNs
            mask = np.isnan(spectra).any(axis=1)
            thetas = thetas[~mask]
            spectra = spectra[~mask]
            num_nan_spectra += mask.sum()

            # Save data for later
            list_of_thetas.append(thetas)
            list_of_spectra.append(spectra)
            wavelengths = np.array(hdf_file["wavelength"])

    print("\nMerging HDF files...", end=" ", flush=True)
    thetas = np.concatenate(list_of_thetas)
    spectra = np.concatenate(list_of_spectra)
    file_path = target_dir / "merged.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        hdf_file.create_dataset(name="theta", data=thetas, dtype=float)
        hdf_file.create_dataset(name="spectra", data=spectra, dtype=float)
        hdf_file.create_dataset(name="wavelengths", data=wavelengths)
    print("Done!\n", flush=True)

    print(f"Total mumber of spectra: {len(spectra):,}", flush=True)
    print(f"Excluded due to NaNs:    {num_nan_spectra:,}\n", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
