"""
Methods for handling the standardization of the data.
"""

from pathlib import Path

import h5py
import numpy as np
import torch

from fm4ar.utils.paths import get_datasets_dir


class Standardizer:
    """
    Simple abstraction for handling the standardization of the data.
    """

    def __init__(
        self,
        flux_mean: float | np.ndarray | torch.Tensor = 0,
        flux_std: float | np.ndarray | torch.Tensor = 1,
        theta_mean: float | np.ndarray | torch.Tensor = 0,
        theta_std: float | np.ndarray | torch.Tensor = 1,
    ) -> None:

        super().__init__()

        if isinstance(flux_mean, float):
            self.flux_mean: float | torch.Tensor = flux_mean
        elif isinstance(flux_mean, np.ndarray):
            self.flux_mean = torch.from_numpy(flux_mean).float()
        else:
            self.flux_mean = flux_mean.float()

        if isinstance(flux_std, float):
            self.flux_std: float | torch.Tensor = flux_std
        elif isinstance(flux_std, np.ndarray):
            self.flux_std = torch.from_numpy(flux_std).float()
        else:
            self.flux_std = flux_std.float()

        if isinstance(theta_mean, float):
            self.theta_mean: float | torch.Tensor = theta_mean
        elif isinstance(theta_mean, np.ndarray):
            self.theta_mean = torch.from_numpy(theta_mean).float()
        else:
            self.theta_mean = theta_mean.float()

        if isinstance(theta_std, float):
            self.theta_std: float | torch.Tensor = theta_std
        elif isinstance(theta_std, np.ndarray):
            self.theta_std = torch.from_numpy(theta_std).float()
        else:
            self.theta_std = theta_std.float()

    def standardize_flux(self, flux: torch.Tensor) -> torch.Tensor:
        return (flux - self.flux_mean) / self.flux_std

    def standardize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean) / self.theta_std

    def inverse_flux(self, flux: torch.Tensor) -> torch.Tensor:
        return flux * self.flux_std + self.flux_mean

    def inverse_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self.theta_std + self.theta_mean


def get_standardizer_from_hdf(
    file_path: Path,
    prefix: str = "",
) -> Standardizer:
    """
    Load the standardization parameters from an HDF file.

    Args:
        file_path: Path to the HDF file containing the standardization
            parameters.
        prefix: Prefix to use for the keys in the HDF file. (Only used
            for the Ardevol Martinez et al. (2022) dataset.)

    Returns:
        Standardizer instance.
    """

    with h5py.File(file_path, "r") as hdf_file:
        return Standardizer(
            flux_mean=np.array(hdf_file[f"{prefix}/flux/mean"]),
            flux_std=np.array(hdf_file[f"{prefix}/flux/std"]),
            theta_mean=np.array(hdf_file[f"{prefix}/theta/mean"]),
            theta_std=np.array(hdf_file[f"{prefix}/theta/std"]),
        )


def get_standardizer(config: dict) -> Standardizer:
    """
    Load the standardization parameters for the given experiment config.
    """

    match name := config["data"]["name"]:

        case "ardevol-martinez-2022":
            file_path = (
                get_datasets_dir()
                / "ardevol-martinez-2022"
                / "precomputed"
                / "standardization_parameters.hdf"
            )
            prefix = str(config["data"]["type"])
            return get_standardizer_from_hdf(file_path, prefix)

        case "goyal-2020":
            raise NotImplementedError()

        case "toy-dataset":
            raise NotImplementedError()

        case "vasist-2023":
            raise NotImplementedError()

        case _:
            raise ValueError(f"Unknown dataset: `{name}`")
