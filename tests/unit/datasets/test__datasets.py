"""
Unit tests for `SpectraDataset` class and `load_dataset` function.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from fm4ar.datasets import load_dataset


@pytest.fixture
def path_to_dataset_1(tmp_path: Path) -> Path:
    """
    Create a dummy dataset (with a single wavelength for all spectra)
    for testing and return the path to it.
    """

    # Create a dummy dataset
    file_path = tmp_path / "dummy_dataset_1.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("theta", data=np.arange(15).reshape(3, 5))
        f.create_dataset("flux", data=np.arange(15).reshape(3, 5))
        f.create_dataset("wlen", data=np.arange(5))

    return file_path


@pytest.fixture
def path_to_dataset_2(tmp_path: Path) -> Path:
    """
    Create a dummy dataset (with different wavelengths for all spectra)
    for testing and return the path to it.
    """

    # Create a dummy dataset
    file_path = tmp_path / "dummy_dataset_2.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("theta", data=np.arange(15).reshape(3, 5))
        f.create_dataset("flux", data=np.arange(15).reshape(3, 5))
        f.create_dataset("wlen", data=np.arange(15).reshape(3, 5))

    return file_path


def test__load_dataset_1(path_to_dataset_1: Path) -> None:
    """
    Unit test for `fm4ar.datasets.load_dataset`.

    This test checks if the function can load a dataset with a single
    wavelength for all spectra.
    """

    # Create the configuration
    config = {
        "dataset": {
            "file_path": path_to_dataset_1,
            "n_samples": 2,
        }
    }

    # Load the dataset
    dataset = load_dataset(config)

    # Basic check of the dataset
    assert len(dataset) == 2
    assert dataset.theta.shape == torch.Size([2, 5])
    assert dataset.flux.shape == torch.Size([2, 5])
    assert dataset.wlen.shape == torch.Size([1, 5])
    assert isinstance(dataset[0], dict)
    assert sorted(dataset[0].keys()) == ["flux", "theta", "wlen"]
    assert dataset[0]["theta"].shape == torch.Size([5])
    assert dataset[0]["flux"].shape == torch.Size([5])
    assert dataset[0]["wlen"].shape == torch.Size([5])


def test__load_dataset_2(path_to_dataset_2: Path) -> None:
    """
    Unit test for `fm4ar.datasets.load_dataset`.

    This test checks if the function can load a dataset with different
    wavelengths for all spectra.
    """

    # Create the configuration
    config = {
        "dataset": {
            "file_path": path_to_dataset_2,
            "n_samples": None,
        }
    }

    # Load the dataset
    dataset = load_dataset(config)

    # Basic check of the dataset
    assert len(dataset) == 3
    assert dataset.theta.shape == torch.Size([3, 5])
    assert dataset.flux.shape == torch.Size([3, 5])
    assert dataset.wlen.shape == torch.Size([3, 5])
    assert isinstance(dataset[0], dict)
    assert sorted(dataset[0].keys()) == ["flux", "theta", "wlen"]
    assert dataset[0]["theta"].shape == torch.Size([5])
    assert dataset[0]["flux"].shape == torch.Size([5])
    assert dataset[0]["wlen"].shape == torch.Size([5])
