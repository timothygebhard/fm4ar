"""
Precompute PCA components for a given dataset.
"""

from argparse import ArgumentParser
from pathlib import Path
from time import time

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA

from fm4ar.utils.paths import get_datasets_dir


def load_spectra(dataset: str) -> tuple[np.ndarray, Path]:
    """
    Load the spectra for the given dataset.
    """

    match dataset:

        case "ardevol-martinez-2022-type-1":
            file_path = (
                get_datasets_dir()
                / "ardevol-martinez-2022"
                / "training"
                / "trans_type1.npy"
            )
            spectra = np.load(file_path.as_posix())
            dataset_dir = get_datasets_dir() / "ardevol-martinez-2022"

        case "ardevol-martinez-2022-type-2":
            file_path = (
                get_datasets_dir()
                / "ardevol-martinez-2022"
                / "training"
                / "trans_type2.npy"
            )
            spectra = np.load(file_path.as_posix())
            dataset_dir = get_datasets_dir() / "ardevol-martinez-2022"

        case "vasist-2023":
            file_path = (
                get_datasets_dir() / "vasist-2023" / "training" / "merged.hdf"
            )
            with h5py.File(file_path, "r") as hdf_file:
                spectra = np.array(hdf_file["spectra"])
            dataset_dir = get_datasets_dir() / "vasist-2023"

        case _:
            raise ValueError(f"Unknown dataset: `{dataset}`")

    return np.array(spectra), dataset_dir


if __name__ == "__main__":

    script_start = time()
    print("\nPRECOMPUTE PCA COMPONENTS FOR DATASET\n")

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "ardevol-martinez-2022-type-1",
            "ardevol-martinez-2022-type-2",
            "vasist-2023",
        ],
        help="The name of the dataset.",
    )
    args = parser.parse_args()

    print("Processing dataset:", args.dataset, "\n", flush=True)

    # Load the dataset
    print("Loading dataset...", end=" ", flush=True)
    spectra, dataset_dir = load_spectra(args.dataset)
    print(f"Done! (Shape: {spectra.shape})", flush=True)

    # Compute the PCA components
    # This produces is a matrix of shape (n_components, n_features), that is,
    # the components are stored as rows.
    print("Computing PCA components...", end=" ", flush=True)
    pca = PCA()
    pca.fit(spectra)
    print("Done!", flush=True)

    # Collect the results that we want to save
    results = {
        "components": pca.components_,
        "mean": pca.mean_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "singular_values": pca.singular_values_,
    }

    # Save the results
    print("Saving results...", end=" ", flush=True)
    precomputed_dir = dataset_dir / "precomputed"
    precomputed_dir.mkdir(exist_ok=True)
    file_path = precomputed_dir / f"{args.dataset}.pt"
    torch.save(results, file_path)
    print("Done!", flush=True)

    print(f"\nThis took {time() - script_start:.1f} seconds!\n")
