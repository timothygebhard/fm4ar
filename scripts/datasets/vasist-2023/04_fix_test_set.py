"""
This script fixes the names of the datasets in the HDF files of the
test set. This is necessary to work with the current version of the
data loading code, but ultimately, we should fix that instead of
relying on this script.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


if __name__ == "__main__":

    script_start = time.time()
    print("\nFIX NAMES FOR TEST SET\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        type=Path,
        required=True,
        help="Path to the HDF file to which to apply the fixes.",
    )
    args = parser.parse_args()

    # Open the HDF file
    with h5py.File(args.file_path, "a") as f:

        # Load flux and noise
        flux = np.array(f["flux"])
        noise = np.array(f["noise"])

        # Overwrite the flux dataset with the noisy flux
        # This is what will be loaded by default by the data loading code!
        del f["flux"]
        f.create_dataset("flux", data=flux + noise, dtype=np.float32)

        # Store the raw flux
        f.create_dataset("raw_flux", data=flux, dtype=np.float32)

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
