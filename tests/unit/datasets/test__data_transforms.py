"""
Unit tests for data_transforms.py.
"""

import numpy as np
import pytest

from fm4ar.datasets.data_transforms import (
    AddNoise,
    Subsample,
    get_data_transforms,
)


def test_add_noise() -> None:
    """
    Test `fm4ar.datasets.data_transforms.AddNoise`.
    """

    # Instantiate the transform
    transform = AddNoise(
        random_seed=42,
        complexity=3,
        transform="lambda wlen, flux_error: flux_error",
    )

    # Create a dummy input
    x = {
        "wlen": np.linspace(0.95, 2.45, 100),
        "flux": np.zeros(100),
    }

    # Apply the forward transformation
    y = transform.forward(x)

    # Check that the output has the same shape as the input
    assert y["flux"].shape == x["flux"].shape
    assert y["wlen"].shape == x["wlen"].shape

    # Check that the wavelength array is unchanged
    assert np.allclose(y["wlen"], x["wlen"])

    # Ensure reproducibility (i.e., the same noise is added)
    assert np.isclose(np.sum(y["flux"]), 0.09608717939064015)


def test_subsample() -> None:
    """
    Test `fm4ar.datasets.data_transforms.Subsample`.
    """

    # Instantiate the transform
    transform = Subsample(
        random_seed=42,
        factor=0.5,
    )

    # Create a dummy input
    x = {
        "wlen": np.linspace(0.95, 2.45, 100),
        "flux": np.zeros(100),
        "error_bars": np.zeros(100),
    }

    # Apply the forward transformation
    y = transform.forward(x)

    # Check that the output has the expected shape
    assert y["flux"].shape[0] == 50
    assert y["wlen"].shape[0] == 50

    # Ensure reproducibility (i.e., the same indices are selected)
    assert np.isclose(np.sum(y["wlen"]), 85.86363636363639)


def test_get_data_transforms() -> None:
    """
    Test `fm4ar.datasets.data_transforms.get_data_transforms`.
    """

    # Define the configuration
    stage_config = {
        "data_transforms": [
            {
                "method": "add_noise",
                "kwargs": {
                    "random_seed": 42,
                    "complexity": 3,
                    "transform": "lambda wlen, flux_error: flux_error",
                },
            },
            {
                "method": "subsample",
                "kwargs": {
                    "random_seed": 42,
                    "factor": 0.5,
                },
            },
        ],
    }

    # Get the data transforms
    data_transforms = get_data_transforms(stage_config)

    # Check that the data transforms are as expected
    assert len(data_transforms) == 2
    assert isinstance(data_transforms[0], AddNoise)
    assert isinstance(data_transforms[1], Subsample)

    # Test illegal method
    stage_config = {"data_transforms": [{"method": "illegal_method"}]}
    with pytest.raises(ValueError) as value_error:
        get_data_transforms(stage_config)
    assert "Unknown data transform" in str(value_error.value)
