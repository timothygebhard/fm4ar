"""
Unit tests for feature_scaling submodule.
"""

import numpy as np
import torch
import pytest
from deepdiff import DeepDiff

from fm4ar.datasets.theta_scalers import (
    ThetaScaler,
    IdentityScaler,
    MeanStdScaler,
    MinMaxScaler,
    get_theta_scaler,
)
from fm4ar.datasets.vasist_2023.prior import THETA_0


def test__theta_scalers() -> None:
    """
    Test the theta feature scalers.
    """

    # Define some mock input data
    sample = {"theta": np.array(THETA_0)}

    # Type annotation
    scaler: ThetaScaler

    # Test the identity scaler (default)
    scaler = get_theta_scaler({})
    assert isinstance(scaler, IdentityScaler)
    assert not DeepDiff(scaler.forward(sample), sample)
    assert not DeepDiff(scaler.inverse(sample), sample)

    # Test the identity scaler (explicit)
    scaler = get_theta_scaler({"method": "identity"})
    assert isinstance(scaler, IdentityScaler)

    # Test the MeanStdScaler
    scaler = get_theta_scaler(
        {"method": "mean_std", "kwargs": {"dataset": "vasist-2023"}}
    )
    assert isinstance(scaler, MeanStdScaler)
    transformed = scaler.forward(sample)
    assert transformed["theta"].shape == (16,)
    for key in sample.keys():
        assert np.allclose(scaler.inverse(transformed)[key], sample[key])

    # Test the MinMaxScaler
    scaler = get_theta_scaler(
        {"method": "min_max", "kwargs": {"dataset": "vasist-2023"}}
    )
    assert isinstance(scaler, MinMaxScaler)
    transformed = scaler.forward(sample)
    assert transformed["theta"].shape == (16,)
    assert np.min(transformed["theta"]) >= 0.0
    assert np.max(transformed["theta"]) <= 1.0
    for key in sample.keys():
        assert np.allclose(scaler.inverse(transformed)[key], sample[key])

    # Test invalid method
    with pytest.raises(ValueError) as value_error:
        get_theta_scaler({"method": "invalid"})
    assert "Unknown feature scaling method:" in str(value_error.value)

    # Test invalid dataset
    with pytest.raises(ValueError) as value_error:
        get_theta_scaler(
            {"method": "min_max", "kwargs": {"dataset": "invalid"}}
        )
    assert "Unknown dataset:" in str(value_error.value)
    with pytest.raises(ValueError) as value_error:
        get_theta_scaler(
            {"method": "mean_std", "kwargs": {"dataset": "invalid"}}
        )
    assert "Unknown dataset:" in str(value_error.value)

    # Test the forward_array and inverse_array methods
    scaler = MeanStdScaler(mean=np.ones(16), std=2 * np.ones(16))
    assert np.allclose(scaler.forward_array(THETA_0), (THETA_0 - 1) / 2)
    assert np.allclose(
        scaler.inverse_array(scaler.forward_array(THETA_0)),
        THETA_0,
    )

    # Test the forward_tensor and inverse_tensor methods
    tensor = torch.from_numpy(THETA_0)
    assert np.allclose(scaler.forward_tensor(tensor), (tensor - 1) / 2)
    assert np.allclose(
        scaler.inverse_tensor(scaler.forward_tensor(tensor)),
        tensor,
    )
