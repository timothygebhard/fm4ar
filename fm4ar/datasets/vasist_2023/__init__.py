"""
Methods for loading the dataset from Vasist et al. (2023).
"""

from copy import deepcopy

import h5py
import numpy as np
import torch

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, LABELS
from fm4ar.datasets.dataset import ArDataset
from fm4ar.utils.paths import get_datasets_dir


def load_vasist_2023_dataset(config: dict) -> ArDataset:
    """
    Load the dataset from Vasist et al. (2023).
    """

    # Do not modify the original config
    config = deepcopy(config)

    # Get the subset to load (training or test)
    if "which" in config["data"]:
        which = config["data"].pop("which")
    else:
        which = "training"

    # Load data from HDF file
    dataset_dir = get_datasets_dir() / "vasist-2023" / which
    file_path = dataset_dir / "merged.hdf"
    with h5py.File(file_path, "r") as hdf_file:
        theta = np.array(hdf_file["theta"])
        x = np.array(hdf_file["spectra"])
        wavelengths = np.array(hdf_file["wavelengths"])

    # Define noise levels
    noise_levels = 1.25754e-17 * 1e16

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta),
        x=torch.from_numpy(x),
        names=LABELS,
        ranges=list(zip(LOWER, UPPER, strict=True)),
        noise_levels=noise_levels,
        wavelengths=torch.from_numpy(wavelengths),
        **config["data"],
    )
