"""
Defines a dense residual network module.
"""

import torch
import torch.nn as nn
from glasflow.nflows.nn.nets.resnet import ResidualBlock

from fm4ar.utils.torchutils import get_activation_from_string


class InitialLayerForZeroInputs(nn.Module):
    """
    This layer is essentially equivalent to `nn.Linear(0, output_dim)`,
    but in a way that behaves consistently for CPU and GPU, and does
    not trigger any warnings upon initialization.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.output_dim, device=x.device)


class DenseResidualNet(nn.Module):
    """
    A ``nn.Module`` consisting of a sequence of dense residual blocks,
    which are interleaved with linear resizing layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "elu",
        dropout: float = 0.0,
        batch_norm: bool = True,
        context_features: int | None = None,
    ) -> None:
        """
        Instantiate a DenseResidualNet module.

        Args:
            input_dim: Dimensionality of the network input.
            output_dim: Dimensionality of the network output.
            hidden_dims: Dimensionalities of the hidden layers.
            activation: Activation function used in the residual blocks.
            dropout: Dropout probability for residual blocks used for
                regularization.
            batch_norm: Whether to use batch normalization.
            context_features: Number of additional context features,
                which are provided to the residual blocks via gated
                linear units. If None, no additional context expected.
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        activation_fn = get_activation_from_string(activation)

        # The first layer is a simple linear layer that maps the input to
        # the the size of the first hidden layer. We need some special logic
        # here for the case when the number of input features is zero, which
        # can happen if all inputs are passed as `context`, that is, `x` has
        # shape `(batch_size, 0)`.
        self.initial_layer: nn.Module
        if self.input_dim == 0:
            self.initial_layer = InitialLayerForZeroInputs(hidden_dims[0])
        else:
            self.initial_layer = nn.Linear(self.input_dim, hidden_dims[0])

        # Create a list of residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=context_features,
                    activation=activation_fn,
                    dropout_probability=dropout,
                    use_batch_norm=batch_norm,
                )
                for n in range(self.num_res_blocks)
            ]
        )

        # Create a list of linear layers that resize the output of one hidden
        # layer (residual block) to the input of the next hidden layer. The
        # last linear layer maps the output of the last hidden layer to the
        # output dimensionality. (This is not a separate layer because we want
        # the number of residual blocks and resize layers to be the same.)
        self.resize_layers = nn.ModuleList(
            [
                (
                    nn.Linear(self.hidden_dims[n - 1], self.hidden_dims[n])
                    if self.hidden_dims[n - 1] != self.hidden_dims[n]
                    else nn.Identity()
                )
                for n in range(1, self.num_res_blocks)
            ]
            + [nn.Linear(self.hidden_dims[-1], self.output_dim)]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        """

        x = self.initial_layer(x)

        for residual_block, resize_layer in zip(
            self.residual_blocks,
            self.resize_layers,
            strict=True,
        ):
            x = residual_block(x, context=context)
            x = resize_layer(x)

        return x
