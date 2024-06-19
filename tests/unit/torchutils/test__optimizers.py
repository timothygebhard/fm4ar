"""
Unit tests for `fm4ar.torchutils.optimizers`.
"""

import pytest
import torch

from fm4ar.torchutils.optimizers import (
    OptimizerConfig,
    get_lr,
    get_optimizer_from_config,
)


def test__optimizer_config() -> None:
    """
    Test `fm4ar.torchutils.optimizers.OptimizerConfig`.
    """

    # Case 1: Valid optimizer type
    optimizer_config = OptimizerConfig(
        type="Adam",
        kwargs={"lr": 0.1, "betas": [0.9, 0.95]},
    )
    assert optimizer_config.type == "Adam"
    assert optimizer_config.kwargs == {"lr": 0.1, "betas": [0.9, 0.95]}

    # Case 2: Invalid optimizer type
    with pytest.raises(ValueError) as value_error:
        OptimizerConfig(
            type="invalid",
            kwargs={"lr": 0.1},
        )
    assert "Invalid optimizer type" in str(value_error)


def test__get_optimizer_from_config() -> None:
    """
    Test `get_optimizer_from_kwargs()` and `get_lr()`.
    """

    # Define dummy model parameters
    model_parameters = list(torch.nn.Linear(1, 1).parameters())

    # Case 1: Adam optimizer
    optimizer_config = OptimizerConfig(
        type="Adam",
        kwargs={"lr": 0.1, "betas": [0.9, 0.95]},
    )
    optimizer = get_optimizer_from_config(model_parameters, optimizer_config)
    assert isinstance(optimizer, torch.optim.Adam)
    assert get_lr(optimizer) == [0.1]

    # Case 2: SGD optimizer
    optimizer_config = OptimizerConfig(
        type="SGD",
        kwargs={"lr": 0.1, "momentum": 0.9},
    )
    optimizer = get_optimizer_from_config(model_parameters, optimizer_config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert get_lr(optimizer) == [0.1]
