"""
Unit tests for data_transforms.py.
"""

import numpy as np
import pytest

from fm4ar.datasets.data_transforms import (
    DataTransformConfig,
    AddNoise,
    Subsample,
    get_data_transforms,
)


def test__add_noise() -> None:
    """
    Test `fm4ar.datasets.data_transforms.AddNoise`.
    """

    # Instantiate the transform
    transform = AddNoise(
        config=dict(
            type="DefaultNoiseGenerator",
            kwargs=dict(
                sigma_min=1.0,
                sigma_max=1.0,
                random_seed=42,
            ),
        )
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
    assert np.isclose(np.mean(y["flux"]), -0.03197121324948966)
    assert np.isclose(np.std(y["flux"]), 1.0271166340887257)


def test__subsample() -> None:
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


def test__get_data_transforms() -> None:
    """
    Test `fm4ar.datasets.data_transforms.get_data_transforms`.
    """

    # Define the configuration
    data_transform_configs = [
        DataTransformConfig(
            type="AddNoise",
            kwargs=dict(
                type="DefaultNoiseGenerator",
                kwargs=dict(
                    sigma_min=1.0,
                    sigma_max=1.0,
                    random_seed=42,
                ),
            )
        ),
        DataTransformConfig(
            type="Subsample",
            kwargs={
                "random_seed": 42,
                "factor": 0.5,
            },
        ),
    ]

    # Get the data transforms
    data_transforms = get_data_transforms(data_transform_configs)

    # Check that the data transforms are as expected
    assert len(data_transforms) == 2
    assert isinstance(data_transforms[0], AddNoise)
    assert isinstance(data_transforms[1], Subsample)

    # Test invalid data transform
    data_transform_configs = [
        DataTransformConfig(
            type="ThisMethodDoesNotExist",
            kwargs={},
        ),
    ]
    with pytest.raises(ValueError) as value_error:
        get_data_transforms(data_transform_configs)
    assert "Unknown data transform" in str(value_error.value)
