"""
Methods for loading the dataset from Ardevol Martinez et al. (2022).
"""

from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.datasets.standardization import get_standardizer
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

    # Define shortcuts
    chemistry_model = config["data"]["type"]
    dataset_dir = get_datasets_dir() / "ardevol-martinez-2022" / "train"
    n_samples = config["data"].get("n_samples")

    # Load metadata
    metadata: dict[str, Any] = pd.read_pickle(dataset_dir / "metadata.p")

    # Determine chemistry model; load parameters and fluxes
    file_path = dataset_dir / f"parameters_type{chemistry_model}.npy"
    theta = np.array(np.load(file_path.as_posix()))[:n_samples]
    file_path = dataset_dir / f"trans_type{chemistry_model}.npy"
    flux = np.array(np.load(file_path.as_posix()))[:n_samples]

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

    # Load standardizer (i.e., standardization parameters based on train set)
    standardizer = get_standardizer(config=config)

    # If requested, select only a subset of the parameters
    if config["data"].get("parameters") is not None:
        parameters: list[int] = list(map(int, config["data"]["parameters"]))
        theta = theta[:, parameters]
        names = [names[i] for i in parameters]
        ranges = [ranges[i] for i in parameters]
        if not isinstance(standardizer.theta_mean, float):
            standardizer.theta_mean = standardizer.theta_mean[parameters]
        if not isinstance(standardizer.theta_std, float):
            standardizer.theta_std = standardizer.theta_std[parameters]

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta).float(),
        flux=torch.from_numpy(flux).float(),
        wlen=torch.from_numpy(output_wlen).float(),
        noise_levels=1e-4 * noise_levels,
        noise_floor=5e-4,  # TODO: Check this
        names=names,
        ranges=ranges,
        standardizer=standardizer,
        **config["data"],
    )


def load_ardevol_martinez_2022_test_dataset(config: dict) -> ArDataset:
    """
    Load the test dataset from Ardevol-Martinez et al. (2022).
    """

    # Get instrument and chemistry model
    chemistry_model = config["data"]["type"]
    instrument = config["data"]["instrument"]
    n_samples = config["data"].get("n_samples")

    # Define shortcut
    key = f"{instrument}/{chemistry_model}"

    # Load data from HDF file
    file_path = (
        get_datasets_dir()
        / "ardevol-martinez-2022"
        / "test"
        / "merged.hdf"
    )
    with h5py.File(file_path.as_posix(), "r") as hdf_file:

        # Get parameters, flux, noise realizations, and wavelengths
        theta = np.array(hdf_file[f"{key}/theta"][:n_samples])
        flux = np.array(hdf_file[f"{key}/flux"][:n_samples])
        noise = np.array(hdf_file[f"{key}/noise"][:n_samples])
        wlen = np.array(hdf_file[instrument]["wlen"])

        # Note: We need the *noise levels* because they are an input to the
        # model even if they are not used to add noise to the flux inside the
        # data loader (because we already add noise here; see below)
        if instrument == "NIRSPEC":
            noise_levels: torch.Tensor | float = torch.from_numpy(
                np.array(hdf_file[instrument]["noise_levels"])
            ).float()
        else:
            noise_levels = float(hdf_file[instrument]["noise_levels"])

        # Get parameter names and ranges
        names = hdf_file[key].attrs["names"]
        ranges = hdf_file[key].attrs["ranges"]

    # Add the noise to the flux
    flux += 1e-4 * noise

    # Load standardizer (i.e., standardization parameters based on train set)
    standardizer = get_standardizer(config=config)

    # If requested, select only a subset of the parameters
    if config["data"].get("parameters") is not None:
        parameters: list[int] = list(map(int, config["data"]["parameters"]))
        theta = theta[:, parameters]
        names = [names[i] for i in parameters]
        ranges = [ranges[i] for i in parameters]
        if not isinstance(standardizer.theta_mean, float):
            standardizer.theta_mean = standardizer.theta_mean[parameters]
        if not isinstance(standardizer.theta_std, float):
            standardizer.theta_std = standardizer.theta_std[parameters]

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta).float(),
        flux=torch.from_numpy(flux).float(),
        wlen=torch.from_numpy(wlen).float(),
        noise_levels=1e-4 * noise_levels,
        noise_floor=0.0,
        names=names,
        ranges=ranges,
        standardizer=standardizer,
        **config["data"],
    )
