"""
Unit tests for `fm4ar.utils.shapes`.
"""

import pytest
import torch

from fm4ar.utils.shapes import validate_dims, validate_shape


def test__validate_dims() -> None:
    """
    Test `fm4ar.utils.torchutils.validate_dims()`.
    """

    # Case 1: Correct dimensions (torch)
    x = torch.randn(10, 5)
    validate_dims(x=x, ndim=2)

    # Case 2: Incorrect dimensions (torch)
    x = torch.randn(10, 5)
    with pytest.raises(ValueError) as value_error:
        validate_dims(x=x, ndim=3)
    assert "Expected `x` to have 3 dimensions but found 2!" in str(value_error)

    # Case 3: Correct dimensions (numpy)
    x = torch.randn(10, 5).numpy()
    validate_dims(x=x, ndim=2)

    # Case 4: Incorrect dimensions (numpy)
    x = torch.randn(10, 5).numpy()
    with pytest.raises(ValueError) as value_error:
        validate_dims(x=x, ndim=3)
    assert "Expected `x` to have 3 dimensions but found 2!" in str(value_error)


def test__validate_shape() -> None:
    """
    Test `fm4ar.utils.torchutils.validate_shape()`.
    """

    # Case 1: Correct shapes (torch)
    x = torch.randn(10, 5)
    validate_shape(x=x, shape=(10, 5))

    # Case 2: Incorrect shapes (torch)
    x = torch.randn(10, 5)
    with pytest.raises(ValueError) as value_error:
        validate_shape(x=x, shape=(5, 10))
    msg = "Expected `x` to have shape (5, 10) but found torch.Size([10, 5])!"
    assert msg in str(value_error)

    # Case 3: Correct shapes (numpy)
    x = torch.randn(10, 5).numpy()
    validate_shape(x=x, shape=(10, 5))

    # Case 4: Incorrect shapes (numpy)
    x = torch.randn(10, 5).numpy()
    with pytest.raises(ValueError) as value_error:
        validate_shape(x=x, shape=(5, 10))
    msg = "Expected `x` to have shape (5, 10) but found (10, 5)!"
    assert msg in str(value_error)
