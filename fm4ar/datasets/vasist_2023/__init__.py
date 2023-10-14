"""
Methods for loading the dataset from Vasist et al. (2023).
"""

import h5py
import numpy as np
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.datasets.standardization import get_standardizer
from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, LABELS, SIGMA
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
        theta = np.array(hdf_file["theta"][:n_samples], dtype=np.float32)
        flux = np.array(hdf_file["flux"][:n_samples], dtype=np.float32)
        wlen = np.array(hdf_file["wlen"], dtype=np.float32)

    # Load standardization parameters
    standardizer = get_standardizer(config)

    # If requested, select only a subset of the parameters
    if config["data"].get("parameters") is not None:
        parameters: list[int] = list(map(int, config["data"]["parameters"]))
        theta = theta[:, parameters]
        names = [LABELS[i] for i in parameters]
        ranges = [(LOWER[i], UPPER[i]) for i in parameters]
    else:
        names = LABELS
        ranges = list(zip(LOWER, UPPER, strict=True))

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta),
        flux=torch.from_numpy(flux),
        wlen=torch.from_numpy(wlen),
        noise_levels=SIGMA,
        noise_floor=0.0,
        names=names,
        ranges=ranges,
        standardizer=standardizer,
        standardize_theta=config["data"].get("standardize_theta", True),
        standardize_flux=config["data"].get("standardize_flux", False),
        add_noise_to_flux=config["data"].get("add_noise_to_flux", True),
    )
