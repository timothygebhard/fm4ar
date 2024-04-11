"""
Unit tests for `fm4ar.nn.mlp`.
"""

import torch
import torch.nn as nn

from fm4ar.nn.mlp import MLP


def test__mlp() -> None:
    """
    Test `fm4ar.nn.mlp.MLP`.
    """

    # Case 1: Single hidden layer, no dropout or batch norm
    mlp = MLP(
        input_dim=10,
        hidden_dims=[5],
        output_dim=1,
        activation="Tanh",
        dropout=0.0,
    )
    assert isinstance(mlp, nn.Module)
    assert isinstance(mlp.mlp, nn.Sequential)
    assert len(mlp.mlp) == 3
    assert isinstance(mlp.mlp[0], torch.nn.Linear)
    assert isinstance(mlp.mlp[1], torch.nn.Tanh)
    assert isinstance(mlp.mlp[2], torch.nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)

    # Case 2: Multiple hidden layers with dropout and batch norm
    mlp = MLP(
        input_dim=10,
        hidden_dims=[5, 5],
        output_dim=1,
        activation="SiLU",
        dropout=0.5,
        batch_norm=True,
    )
    assert isinstance(mlp, nn.Module)
    assert isinstance(mlp.mlp, torch.nn.Sequential)
    assert len(mlp.mlp) == 9
    assert isinstance(mlp.mlp[0], torch.nn.Linear)
    assert isinstance(mlp.mlp[1], torch.nn.SiLU)
    assert isinstance(mlp.mlp[2], torch.nn.BatchNorm1d)
    assert isinstance(mlp.mlp[3], torch.nn.Dropout)
    assert isinstance(mlp.mlp[4], torch.nn.Linear)
    assert isinstance(mlp.mlp[5], torch.nn.SiLU)
    assert isinstance(mlp.mlp[6], torch.nn.BatchNorm1d)
    assert isinstance(mlp.mlp[7], torch.nn.Dropout)
    assert isinstance(mlp.mlp[8], torch.nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)
