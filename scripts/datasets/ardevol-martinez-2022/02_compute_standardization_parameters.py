"""
Compute the standardization parameters for the dataset.
"""

import time

import h5py
import numpy as np

from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nCOMPUTE STANDARDIZATION PARAMETERS\n")

    # Ensure the output directory exists
    dataset_dir = get_datasets_dir() / "ardevol-martinez-2022"
    train_dir = dataset_dir / "train"
    precomputed_dir = dataset_dir / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    # Create a new HDF file for the standardization parameters
    file_path = precomputed_dir / "standardization_parameters.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        for chem_model in ("1", "2"):

            print(f"Computing parameters for type {chem_model}...", end=" ")

            # Load the training dataset
            data = {
                "theta": np.load(
                    (train_dir / f"parameters_type{chem_model}.npy").as_posix()
                ),
                "flux": np.load(
                    (train_dir / f"trans_type{chem_model}.npy").as_posix()
                ),
            }

            # Compute the mean and std
            for key, values in data.items():
                hdf_file.create_dataset(
                    name=f"{chem_model}/{key}/mean",
                    data=np.mean(values, axis=0),
                    dtype=float,
                )
                hdf_file.create_dataset(
                    name=f"{chem_model}/{key}/std",
                    data=np.std(values, axis=0, ddof=1),  # matches torch.std
                    dtype=float,
                )

            print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
