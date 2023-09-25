"""
Methods for loading the dataset from Vasist et al. (2023).
"""

import h5py
import numpy as np
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.datasets.standardization import get_standardizer
from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, LABELS
from fm4ar.utils.paths import get_datasets_dir


def load_vasist_2023_dataset(config: dict) -> ArDataset:
    """
    Load the dataset from Vasist et al. (2023).
    """

    # Define shortcuts
    which = config["data"]["which"]
    n_samples = config["data"].get("n_samples")
    file_name = config["data"].get("file_name", "merged.hdf")

    # Load data from HDF file
    dataset_dir = get_datasets_dir() / "vasist-2023" / which
    file_path = dataset_dir / file_name
    with h5py.File(file_path, "r") as hdf_file:
        theta = np.array(hdf_file["theta"][:n_samples])
        flux = np.array(hdf_file["spectra"][:n_samples])
        wlen = np.array(hdf_file["wavelengths"])

    # Define noise levels
    noise_levels = 1.25754e-17 * 1e16

    # Load standardization parameters
    standardizer = get_standardizer(config)

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta),
        flux=torch.from_numpy(flux),
        wlen=torch.from_numpy(wlen),
        noise_levels=noise_levels,
        noise_floor=0.0,
        names=LABELS,
        ranges=list(zip(LOWER, UPPER, strict=True)),
        standardizer=standardizer,
        **config["data"],
    )
