"""
Methods for loading the dataset from Ardevol Martinez et al. (2022).
"""

from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from spectres import spectres

from fm4ar.datasets.dataset import ArDataset
from fm4ar.utils.paths import get_datasets_dir


def load_ardevol_martinez_2022_dataset(config: dict) -> ArDataset:
    """
    Load the dataset from Ardevol-Martinez et al. (2022).
    """

    # Get the subset to load (training or test)
    which = config["data"].pop("which", "training")

    # Load the dataset
    if which == "training":
        return load_ardevol_martinez_2022_training_dataset(config=config)
    elif which == "test":
        return load_ardevol_martinez_2022_test_dataset(config=config)

    raise ValueError(f"Unknown subset: `{which}`")


def load_ardevol_martinez_2022_training_dataset(config: dict) -> ArDataset:
    """
    Load the training dataset from Ardevol-Martinez et al. (2022).
    """

    dataset_dir = get_datasets_dir() / "ardevol-martinez-2022" / "training"

    # Load metadata
    metadata: dict[str, Any] = pd.read_pickle(dataset_dir / "metadata.p")

    # Load data
    if config["data"]["type"] == "type-1":
        file_path_theta = dataset_dir / "parameters_type1.npy"
        file_path_x = dataset_dir / "trans_type1.npy"
        chemistry_model = 1
    elif config["data"]["type"] == "type-2":
        file_path_theta = dataset_dir / "parameters_type2.npy"
        file_path_x = dataset_dir / "trans_type2.npy"
        chemistry_model = 2
    else:
        raise ValueError(f"Unknown type: `{config['data']['type']}`")
    theta = torch.from_numpy(np.load(file_path_theta.as_posix()))
    x = torch.from_numpy(np.load(file_path_x.as_posix()))

    # Select input and output wavelengths
    input_wavelengths = metadata["Wavelength"]["NIRSPEC"]
    match (instrument := config["data"].pop("instrument")):
        case "NIRSPEC":
            output_wavelengths = metadata["Wavelength"]["NIRSPEC"]
        case "WFC3":
            output_wavelengths = metadata["Wavelength"]["WFC3"]
        case _:
            raise ValueError(f"Unknown instrument: `{instrument}`")

    # Resample spectra to target wavelengths
    x = torch.from_numpy(
        spectres(
            new_wavs=output_wavelengths,
            spec_wavs=input_wavelengths,
            spec_fluxes=x.numpy(),
        )
    )

    # Load noise levels
    noise_levels: float | torch.Tensor
    if instrument == "NIRSPEC":
        noise_levels = torch.from_numpy(metadata["Noise"]["NIRSPEC"])
    else:
        noise_levels = float(metadata["Noise"]["WFC3"])

    # Create dataset
    return ArDataset(
        theta=theta,
        x=x,
        wavelengths=torch.from_numpy(output_wavelengths).float(),
        noise_levels=1e-4 * noise_levels,  # TODO: check this
        names=metadata["names"][chemistry_model],
        ranges=metadata["ranges"][chemistry_model],
        **config["data"],
    )


def load_ardevol_martinez_2022_test_dataset(config: dict) -> ArDataset:
    """
    Load the test dataset from Ardevol-Martinez et al. (2022).
    """

    # Get instrument and chemistry model
    instrument = config["data"]["instrument"]
    chemistry_model = config["data"]["type"].split("-")[1]

    # Load data from HDF file
    file_path = (
        get_datasets_dir()
        / "ardevol-martinez-2022"
        / "test"
        / "merged.hdf"
    )
    with h5py.File(file_path.as_posix(), "r") as hdf_file:
        theta = np.array(hdf_file[instrument][chemistry_model]["theta"])
        spectra = np.array(hdf_file[instrument][chemistry_model]["spectra"])
        wavelengths = np.array(hdf_file[instrument]["wavelengths"])
        _noise_levels = np.array(hdf_file[instrument]["noise_levels"])
        names = hdf_file[instrument][chemistry_model].attrs["names"]
        ranges = hdf_file[instrument][chemistry_model].attrs["ranges"]

    # Convert noise levels to torch.Tensor or float
    noise_levels: float | torch.Tensor
    if _noise_levels.ndim == 0:
        noise_levels = float(_noise_levels)
    else:
        noise_levels = torch.from_numpy(_noise_levels).float()

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta).float(),
        x=torch.from_numpy(spectra).float(),
        wavelengths=torch.from_numpy(wavelengths).float(),
        noise_levels=noise_levels,
        names=names,
        ranges=ranges,
        **config["data"],
    )
