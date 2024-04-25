"""
Defines a dense residual network module.
"""

import torch
import torch.nn as nn

from fm4ar.torchutils.general import get_activation_from_name


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


class ResidualBlock(nn.Module):
    """
    A general-purpose residual block.
    Originally based on `glasflow.nflows.nn.nets.resnets.ResidualBlock`.
    """

    def __init__(
        self,
        n_features: int,
        n_context_features: int | None,
        activation: nn.Module,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):

        super().__init__()

        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_probability)

        if use_batch_norm and use_layer_norm:  # pragma: no cover
            raise ValueError(
                "Cannot use both batch normalization and layer normalization!"
            )

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(n_features, eps=1e-3)
            self.batch_norm_2 = nn.BatchNorm1d(n_features, eps=1e-3)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm_1 = nn.LayerNorm(n_features)
            self.layer_norm_2 = nn.LayerNorm(n_features)

        if n_context_features is not None:
            self.context_layer = nn.Linear(n_context_features, n_features)

        self.linear_layer_1 = nn.Linear(n_features, n_features)
        self.linear_layer_2 = nn.Linear(n_features, n_features)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> torch.Tensor:

        temps = inputs

        if self.use_batch_norm:
            temps = self.batch_norm_1(temps)
        if self.use_layer_norm:
            temps = self.layer_norm_1(temps)
        temps = self.activation(temps)
        temps = self.linear_layer_1(temps)

        if self.use_batch_norm:
            temps = self.batch_norm_2(temps)
        if self.use_layer_norm:
            temps = self.layer_norm_2(temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layer_2(temps)

        if context is not None:
            temps = torch.nn.functional.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1,
            )

        return inputs + temps


class DenseResidualNet(nn.Module):
    """
    A ``nn.Module`` consisting of a sequence of dense residual blocks,
    which are interleaved with linear resizing layers.
    """

    # This class can be used as a block inside an embedding network. For this
    # reason, we need the following flag so that `create_embedding_net_block`
    # knows it needs to pass the input shape to the constructor.
    requires_input_shape = True

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "ELU",
        final_activation: str | None = None,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        context_features: int | None = None,
    ) -> None:
        """
        Instantiate a DenseResidualNet module.

        Args:
            input_shape: Shape of the input tensor. Note: This must be
                a tuple of length 1: multi-dimensional inputs are not
                supported. We use `input_shape` instead of `input_dim`
                for compatibility.
            output_dim: Dimensionality of the network output.
            hidden_dims: Dimensionalities of the hidden layers.
            activation: Activation function used in the residual blocks.
            final_activation: Activation function used after the final
                layer. If None, no activation is used (default).
            dropout: Dropout probability for residual blocks used for
                regularization.
            use_batch_norm: Whether to use batch normalization.
            use_layer_norm: Whether to use layer normalization.
            context_features: Number of additional context features,
                which are provided to the residual blocks via gated
                linear units. If None, no additional context expected.
        """

        super().__init__()

        # Check if the input shape is valid
        if len(input_shape) != 1:
            raise ValueError(
                "DenseResidualNet only supports 1D inputs! "
                f"Got input shape: {input_shape}"
            )

        self.input_dim = input_shape[0]
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

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
        # NOTE: We don't assign `get_activation_from_string(activation)` to a
        # variable because we want to make sure that the activation function is
        # a separate object in each block (so that we can change the parameters
        # of each block independently).
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    n_features=self.hidden_dims[n],
                    n_context_features=context_features,
                    activation=get_activation_from_name(activation),
                    dropout_probability=dropout,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
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

        # Define final activation function
        # The default usecase for this is to use a sigmoid activation function
        # in case the target parameters have been scaled to the range [0, 1]
        if final_activation is not None:
            self.final_activation = get_activation_from_name(final_activation)
        else:
            self.final_activation = nn.Identity()

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

        x = self.final_activation(x)

        return x
