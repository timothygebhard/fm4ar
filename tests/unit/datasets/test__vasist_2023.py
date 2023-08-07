"""
Test loading the Vasist-2023 dataset.
"""

import pytest

from fm4ar.datasets import load_dataset
from fm4ar.utils.paths import get_datasets_dir


VASIST_2023_DIR = get_datasets_dir() / "vasist-2023"


@pytest.mark.skipif(
    condition=not (VASIST_2023_DIR / "training" / "merged.hdf").exists(),
    reason="Vasist-2023 training dataset is not available!"
)
def test__load_vasist_2023_training_dataset() -> None:
    """
    Test loading the Vasist-2023 training dataset.
    """

    # Define configuration
    config = {
        "data": {
            "name": "vasist-2023",
            "which": "training",
        }
    }

    # Load the dataset
    dataset = load_dataset(config=config)

    # Check that the dataset is not empty
    assert len(dataset) > 0


@pytest.mark.skipif(
    condition=not (VASIST_2023_DIR / "test" / "merged.hdf").exists(),
    reason="Vasist-2023 test dataset is not available!"
)
def test__load_vasist_2023_test_dataset() -> None:

    # Define configuration and load the dataset
    config = {
        "data": {
            "name": "vasist-2023",
            "which": "test",
            "n_samples": 100,
        }
    }
    dataset = load_dataset(config=config)

    # Basic checks
    assert len(dataset) == 100
    assert dataset.theta_dim == 16
    assert dataset.context_dim == (947, )
    theta, x = dataset[0]
    assert theta.shape == (16, )
    assert x.shape == (947, )

    # Check that we can also load the wavelengths
    config["data"]["return_wavelengths"] = True
    dataset = load_dataset(config=config)
    assert dataset.context_dim == (947, 2)
    theta, x = dataset[0]
    assert theta.shape == (16, )
    assert x.shape == (947, 2)
