"""
Defines a function to build a multi-layer perceptron (MLP) in PyTorch.
"""

from typing import Sequence

import torch
from torch import nn as nn

from fm4ar.torchutils.general import get_activation_from_name


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str,
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """
        Instantiate an MLP with the given parameters.

        Args:
            input_dim: Dimension of the input.
            hidden_dims: List of hidden dimensions.
            output_dim: Dimension of the output.
            activation: Name of the activation function.
            batch_norm: Whether to use batch normalization.
            layer_norm: Whether to use layer normalization.
            dropout: Dropout probability (between 0 and 1).

        Returns:
            A multi-layer perceptron with the given parameters.
        """

        super().__init__()

        # Prepare list of layers
        layers = torch.nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        # Sanity check: Can't use both batch and layer normalization
        if batch_norm and layer_norm:
            raise ValueError("Can't use both batch and layer normalization.")

        # Note: There seems to be no clear consensus about the order of the
        # activation function and the batch normalization layer.
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(get_activation_from_name(activation))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
                if layer_norm:
                    layers.append(torch.nn.LayerNorm(dims[i + 1]))
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        """

        return torch.Tensor(self.mlp(x))
