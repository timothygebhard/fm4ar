"""
Methods for loading the dataset from Ardevol Martinez et al. (2022).
"""

from copy import deepcopy

import h5py
import numpy as np
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.utils.paths import get_datasets_dir


def load_goyal_2020_dataset(config: dict) -> ArDataset:
    """
    Load the dataset based on Goyal et al. (2020).
    """

    # Do not modify the original config
    config = deepcopy(config)

    # Get the subset to load (training or test)
    which = config["data"].pop("which", "training")
    which = "train" if which == "training" else which

    # Load data from HDF file
    dataset_dir = get_datasets_dir() / "goyal-2020" / "output"
    file_path = dataset_dir / f"{which}.hdf"
    with h5py.File(file_path, "r") as hdf_file:
        names = [str(name) for name in hdf_file.attrs["names"]]
        ranges = [(float(x[0]), float(x[1])) for x in hdf_file.attrs["ranges"]]
        if (n_samples := config["data"].get("n_samples")) is None:
            theta = np.array(hdf_file["theta"])
            flux = np.array(hdf_file["flux"])
            wlen = np.array(hdf_file["wlen"])
        else:
            theta = np.array(hdf_file["theta"][:n_samples])
            flux = np.array(hdf_file["flux"][:n_samples])
            wlen = np.array(hdf_file["wlen"])

    # Define noise levels
    # TODO: What is a good choice here?
    # TODO: Can we convert this to an SNR?
    noise_levels = 0.005

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta),
        x=torch.from_numpy(flux),
        wavelengths=torch.from_numpy(wlen),
        noise_levels=noise_levels,
        names=names,
        ranges=ranges,
        **config["data"],
    )
