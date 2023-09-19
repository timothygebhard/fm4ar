"""
Methods for loading the dataset from Ardevol Martinez et al. (2022).
"""

from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.utils.paths import get_datasets_dir
from fm4ar.utils.resampling import resample_spectrum


def load_ardevol_martinez_2022_dataset(config: dict) -> ArDataset:
    """
    Load the dataset from Ardevol-Martinez et al. (2022).
    """

    if (which := config["data"]["which"]) == "train":
        return load_ardevol_martinez_2022_train_dataset(config=config)
    elif which == "test":
        return load_ardevol_martinez_2022_test_dataset(config=config)

    raise ValueError(f"Unknown subset: `{which}`")


def load_ardevol_martinez_2022_train_dataset(config: dict) -> ArDataset:
    """
    Load the train dataset from Ardevol-Martinez et al. (2022).
    """

    dataset_dir = get_datasets_dir() / "ardevol-martinez-2022" / "train"

    # Load metadata
    metadata: dict[str, Any] = pd.read_pickle(dataset_dir / "metadata.p")

    # Determine chemistry model; load parameters and fluxes
    chemistry_model = config["data"]["type"]
    file_path = dataset_dir / f"parameters_type{chemistry_model}.npy"
    theta = np.array(np.load(file_path.as_posix()))
    file_path = dataset_dir / f"trans_type{chemistry_model}.npy"
    flux = np.array(np.load(file_path.as_posix()))

    # Define parameter names and ranges
    names = metadata["names"][chemistry_model]
    ranges = metadata["ranges"][chemistry_model]

    # Select input and output wavelengths
    input_wlen = np.array(metadata["Wavelength"]["NIRSPEC"])
    instrument = config["data"].pop("instrument")
    if instrument == "NIRSPEC":
        output_wlen = np.array(metadata["Wavelength"]["NIRSPEC"])
    elif instrument == "WFC3":
        output_wlen = np.array(metadata["Wavelength"]["WFC3"])
    else:
        raise ValueError(f"Unknown instrument: `{instrument}`")

    # Resample spectra to target wavelengths, if needed
    if not np.array_equal(input_wlen, output_wlen):
        output_wlen, flux = resample_spectrum(
            new_wlen=output_wlen,
            old_wlen=input_wlen,
            old_flux=flux,
        )

    # Load noise levels
    noise_levels: float | torch.Tensor
    if instrument == "NIRSPEC":
        noise_levels = torch.from_numpy(metadata["Noise"]["NIRSPEC"]).float()
    else:
        noise_levels = float(metadata["Noise"]["WFC3"])

    # If requested, select only a subset of the parameters
    if config["data"].get("parameters") is not None:
        parameters = np.array(config["data"]["parameters"], dtype=int)
        theta = theta[:, parameters]
        names = [names[i] for i in parameters]
        ranges = [ranges[i] for i in parameters]

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta).float(),
        flux=torch.from_numpy(flux).float(),
        wlen=torch.from_numpy(output_wlen).float(),
        noise_levels=1e-4 * noise_levels,
        noise_floor=5e-4,  # TODO: Check this
        names=names,
        ranges=ranges,
        **config["data"],
    )


def load_ardevol_martinez_2022_test_dataset(config: dict) -> ArDataset:
    """
    Load the test dataset from Ardevol-Martinez et al. (2022).
    """

    # Get instrument and chemistry model
    instrument = config["data"]["instrument"]
    chemistry_model = config["data"]["type"]

    # Load data from HDF file
    file_path = (
        get_datasets_dir()
        / "ardevol-martinez-2022"
        / "test"
        / "merged.hdf"
    )
    with h5py.File(file_path.as_posix(), "r") as hdf_file:
        theta = np.array(hdf_file[instrument][chemistry_model]["theta"])
        flux = np.array(hdf_file[instrument][chemistry_model]["flux"])
        noise = np.array(hdf_file[instrument][chemistry_model]["noise"])
        wlen = np.array(hdf_file[instrument]["wlen"])
        names = hdf_file[instrument][chemistry_model].attrs["names"]
        ranges = hdf_file[instrument][chemistry_model].attrs["ranges"]

    # Add the noise to the flux
    flux += 1e-4 * noise

    # If requested, select only a subset of the parameters
    if config["data"].get("parameters") is not None:
        parameters = np.array(config["data"]["parameters"], dtype=int)
        theta = theta[:, parameters]
        names = [names[i] for i in parameters]
        ranges = [ranges[i] for i in parameters]

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta).float(),
        flux=torch.from_numpy(flux).float(),
        wlen=torch.from_numpy(wlen).float(),
        noise_levels=0.0,  # noise was already added to the flux!
        noise_floor=0.0,
        names=names,
        ranges=ranges,
        **config["data"],
    )
