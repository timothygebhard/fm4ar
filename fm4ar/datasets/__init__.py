"""
Load a dataset from the given experiment configuration.
"""

from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, Field

from fm4ar.datasets.dataset import SpectraDataset
from fm4ar.datasets.theta_scalers import (
    MeanStdScaler,
    MinMaxScaler,
    get_theta_scaler,
)
from fm4ar.utils.paths import expand_env_variables_in_path


class DatasetConfig(BaseModel):
    """
    Configuration for the dataset.
    """

    file_path: Path = Field(
        ...,
        description="Path to the HDF5 file containing the dataset.",
    )
    n_train_samples: int = Field(
        ...,
        description="Number of samples to use for training.",
    )
    n_valid_samples: int = Field(
        ...,
        description="Number of samples to use for validation.",
    )
    random_seed: int = Field(
        default=42,
        description=(
            "Random seed for the data loaders: This seed controls how the "
            "dataset is split into training and validation sets. We want to "
            "be able to control this independently of, for example, the "
            "initialization of the model weights."
        ),
    )
    parameters: list[int] | None = Field(
        None,
        description=(
            "Binary mask indicating which parameters to use. "
            "If None, all parameters are used (default)."
        ),
    )


def load_dataset(config: dict) -> SpectraDataset:
    """
    Load a dataset from the given experiment configuration.
    """

    # Extract and very the dataset configuration
    dataset_config = DatasetConfig(**config["dataset"])

    # Get the path to the dataset file
    # The expand_env_variables_in_path() allows to specify the path using
    # environment variables, e.g., $FM4AR_DATASETS_DIR
    file_path = expand_env_variables_in_path(dataset_config.file_path)

    # Determine the total number of samples that we need to load
    n_samples = dataset_config.n_train_samples + dataset_config.n_valid_samples

    # Load the dataset
    with h5py.File(file_path, "r") as f:
        theta = np.array(f["theta"][:n_samples])
        flux = np.array(f["flux"][:n_samples])
        wlen = np.array(
            f["wlen"] if len(f["wlen"].shape) == 1 else f["wlen"][:n_samples]
        )

    # Select only the parameters that we want to use
    if dataset_config.parameters is not None:
        if len(dataset_config.parameters) != theta.shape[1]:
            raise ValueError(  # pragma: no cover
                "The number of parameters in the dataset does not match the "
                "number of parameters specified in the configuration!"
            )
        mask = np.array(dataset_config.parameters, dtype=bool)
        theta = theta[:, mask]
    else:
        mask = np.ones(theta.shape[1], dtype=bool)

    # TODO: Add support for filtering the dataset, e.g., based on mean flux

    # Ensure that wlen is 2D
    if wlen.ndim == 1:
        wlen = wlen[None, :]

    # Make sure the lengths match
    if theta.shape[0] != flux.shape[0]:
        raise ValueError(  # pragma: no cover
            "The number of samples does not match between `theta` and `flux`!"
        )
    if wlen.shape[0] != 1 and wlen.shape[0] != flux.shape[0]:
        raise ValueError(  # pragma: no cover
            "The number of samples does not match between `wlen` and `flux`! "
            "`wlen` should have either the same length as `flux` or a single "
            "wavelength for all spectra."
        )
    if wlen.shape[1] != flux.shape[1]:
        raise ValueError(  # pragma: no cover
            "The number of bins does not match between `wlen` and `flux`!"
        )

    # Construct the feature scaling transforms
    theta_scaler_config = config.get("theta_scaler", {})
    theta_scaler = get_theta_scaler(theta_scaler_config)

    # Apply the parameter mask also to the theta scaler, if necessary
    if isinstance(theta_scaler, MeanStdScaler):
        theta_scaler.mean = theta_scaler.mean[mask]
        theta_scaler.std = theta_scaler.std[mask]
    elif isinstance(theta_scaler, MinMaxScaler):
        theta_scaler.minimum = theta_scaler.minimum[mask]
        theta_scaler.maximum = theta_scaler.maximum[mask]

    # Construct the dataset with the theta scaler
    dataset = SpectraDataset(
        theta=theta,
        flux=flux,
        wlen=wlen,
        theta_scaler=theta_scaler,
    )

    return dataset
