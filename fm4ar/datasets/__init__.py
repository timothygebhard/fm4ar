"""
Load a dataset from the given experiment configuration.
"""

from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, Field

from fm4ar.datasets.dataset import SpectraDataset
from fm4ar.datasets.theta_scalers import get_theta_scaler
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
            f["wlen"]
            if len(f["wlen"].shape) == 1
            else f["wlen"][:n_samples]
        )

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

    # Construct the dataset with the theta scaler
    dataset = SpectraDataset(
        theta=theta,
        flux=flux,
        wlen=wlen,
        theta_scaler=theta_scaler,
    )

    return dataset
