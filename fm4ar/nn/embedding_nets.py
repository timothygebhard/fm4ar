"""
Different embedding networks and convenience functions.
"""

from os.path import expandvars
from math import pi
from typing import Any
from warnings import catch_warnings, filterwarnings

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from fm4ar.nn.modules import Rescale, Unsqueeze, Tile, Mean
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.utils.torchutils import load_and_or_freeze_model_weights


def create_embedding_net(
    input_dim: tuple[int, ...],
    embedding_net_kwargs: dict,
) -> tuple[nn.Module, int]:
    """
    Create an embedding net from the given keyword arguments.

    Note: An embedding network can consist of multiple parts. For
    example, the first stage might be a single linear layer with
    weights that have been pre-computed from a PCA, followed by a
    second stage that is a standard neural network.

    Therefore, the `embedding_net_kwargs` should be a dictionary with
    the configurations of the different stages. Its keys should be
    the names of the stages (which can be abitrary), and the values
    should be dictionaries with the keyword arguments for the network
    of that stage.

    Args:
        input_dim: The dimension of the input to be embedded.
        embedding_net_kwargs: The configuration of the embedding net.

    Returns:
        A 2-tuple, `(embedding_net, output_dim)`, consisting of the
        embedding network and the dimension of the embedding space.
    """

    # Get parameters for freezing weights, or loading weights from a file
    freeze_weights = embedding_net_kwargs.pop("freeze_weights", False)
    load_weights = embedding_net_kwargs.pop("load_weights", {})

    # Create and collect the embedding net stages
    stages = nn.ModuleDict()
    for stage_name, stage_kwargs in embedding_net_kwargs.items():

        # Create the stage and determine the output dimension
        stages[stage_name], output_dim = create_embedding_net_stage(
            input_dim=input_dim,
            embedding_net_stage_kwargs=stage_kwargs,
        )

        # Update input_dim: The output dimension of the current stage is
        # the input dimension of the next stage
        input_dim = output_dim

    # Create the embedding net: In case we got an empty kwargs dictionary,
    # we simply return an identity function.
    embedding_net: nn.Module
    if not stages:
        embedding_net = nn.Identity()
    else:
        embedding_net = nn.Sequential()
        for name, stage in stages.items():
            embedding_net.add_module(name, stage)

    # Ensure that the final output dimension is 1-dimensional (n-dimensional
    # embeddings are currently not supported). We check against `input_dim`
    # because this always contains the output dimension of the last stage,
    # even if the embedding net is empty (identity).
    if len(input_dim) != 1:
        raise ValueError(
            "The final output dimension of the embedding net "
            f"should be 1, but is {len(input_dim)}!"
        )
    else:
        final_output_dim = int(input_dim[0])

    # Load pre-trained weights or freeze the weights of the flow
    load_and_or_freeze_model_weights(
        model=embedding_net,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    return embedding_net, final_output_dim


def create_embedding_net_stage(
    input_dim: tuple[int, ...],
    embedding_net_stage_kwargs: dict,
) -> tuple[nn.Module, tuple[int, ...]]:
    """
    Create an embedding net stage from the given keyword arguments.

    Args:
        input_dim: The dimension of the input to the stage.
        embedding_net_stage_kwargs: The configuration of the embedding
            net stage. This should have the following keys:

    Returns:
        A 2-tuple, `(stage, output_dim)`, consisting of the embedding
        net stage and the shape of the output of the stage.
    """

    model_type = embedding_net_stage_kwargs["model_type"]
    kwargs = embedding_net_stage_kwargs["kwargs"]

    # Create the stage
    stage: nn.Module
    match model_type:
        case "DenseResidualNet":
            if len(input_dim) != 1:
                raise ValueError("DenseResidualNet only supports 1D inputs!")
            else:
                input_dim_as_int = int(input_dim[0])
            stage = DenseResidualNet(input_dim=input_dim_as_int, **kwargs)
        case "PositionalEncoding":
            stage = PositionalEncoding(**kwargs)
        case "PrecomputedPCAEmbedding":
            stage = PrecomputedPCAEmbedding(**kwargs)
        case "SoftClip":
            stage = SoftClip(**kwargs)
        case "CNPEncoder":
            stage = CNPEncoder(**kwargs)
        case "TransformerEmbedding":
            stage = TransformerEmbedding(**kwargs)
        case _:
            raise ValueError(f"Invalid model type: {model_type}!")

    # Create some dummy input to determine the output dimension of the stage
    batch_size = 19
    dummy_input = torch.zeros((batch_size, *input_dim))
    output_dim = stage(dummy_input).shape[1:]

    return stage, output_dim


class PositionalEncoding(nn.Module):
    """
    A positional encoding module that can be used to encode the time
    and/or the target parameters.
    """

    def __init__(
        self,
        n_freqs: int,
        encode_theta: bool = True,
        base_freq: float = 2 * pi,
    ) -> None:
        """
        Instantiate a `PositionalEncoding` object.

        Args:
            n_freqs: Number of frequencies to use.
            encode_theta: If True, the target parameters are encoded.
                If False, only the time is encoded.
            base_freq: The base frequency. The frequencies used are
                `base_freq * 2^k`, with `k = 0, ..., n_freqs - 1`.
        """

        super(PositionalEncoding, self).__init__()

        self.n_freqs = n_freqs
        self.encode_theta = encode_theta
        self.base_freq = base_freq

        # Create the frequencies and register them as a buffer
        freqs = (base_freq * 2 ** torch.arange(0, n_freqs)).view(1, 1, n_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, t_theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the positional encoding module.
        """

        batch_size = t_theta.shape[0]

        # `t_theta` has shape (batch_size, 1 + theta_dim): The first column
        # contains the time, and the remaining columns contain the parameters.
        if self.encode_theta:
            x = t_theta.view(batch_size, -1, 1) * self.freqs
        else:
            x = t_theta[:, 0:1].view(batch_size, 1, 1) * self.freqs

        # After selecting and reshaping, `x` now has shape:
        # (batch_size, 1 + int(encode_theta) * theta_dim, n_freqs)

        # Apply the positional encoding and flatten the frequency dimension.
        # The shape of the output is:
        # (batch_size, (1 + int(encode_theta) * theta_dim) * n_freqs)
        cos_enc = torch.cos(x).view(batch_size, -1)
        sin_enc = torch.sin(x).view(batch_size, -1)

        # Stack together the original input and the positional encodings.
        # The shape of the output is:
        # (batch_size,
        #  1 + dim_theta + 2 * (1 + int(encode_theta) * theta_dim) * n_freqs)
        encoded = torch.cat((t_theta, cos_enc, sin_enc), dim=1)

        return encoded


class PrecomputedPCAEmbedding(nn.Module):
    """
    An embedding network that consists of a single linear layer with
    weights that have been pre-computed from an SVD.
    """

    def __init__(
        self,
        file_path: str,  # This is called from **kwargs
        n_components: int,
        subtract_mean: bool = True,
        freeze_weights: bool = False,
    ) -> None:
        """
        Instantiate a PrecomputedPCAEmbedding module.

        Args:
            file_path: Path to the file containing the pre-computed
                PCA weights. This should be a `.pt` file containing a
                dictionary with the following keys:
                    - "mean": The mean of the training data.
                    - "components": The PCA components, as a matrix
                        with shape `(n_components, input_size)`, that
                        is, with the components in the rows.
            n_components: The number of PCA components to use.
            subtract_mean: Whether to subtract the mean from the input
                before applying the PCA. (Recommended, unless the data
                loader already does this.)
            freeze_weights: Whether to freeze the weights of the
                embedding network.
        """

        super().__init__()

        self.n_components = n_components
        self.subtract_mean = subtract_mean
        self.freeze_weights = freeze_weights

        # Load the pre-computed PCA weights
        precomputed = torch.load(expandvars(file_path))
        self.components = torch.from_numpy(precomputed["components"]).float()

        # Store the mean as a buffer (we need a buffer here to make sure that
        # the mean is moved to the same device as the model)
        mean = torch.from_numpy(precomputed["mean"]).float()
        self.register_buffer("mean", mean)

        # Determine input size from the PCA components
        self.input_size = self.components.shape[0]

        # Create the linear layer that applies the PCA
        self.linear = nn.Linear(
            in_features=self.input_size,
            out_features=self.n_components,
            bias=False,
        )

        # Initialize the linear layer with the PCA components
        # We transpose the components because the linear layer expects
        # the components to be in the columns, not in the rows.
        self.linear.weight.data = self.components[: self.n_components]

        # Optionally: Freeze the weights
        self.linear.weight.requires_grad_(not self.freeze_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network.
        """

        # Optionally: Subtract the mean
        if self.subtract_mean:
            x = x - self.mean

        # Apply the PCA
        x = self.linear(x)

        return x


class SoftClip(nn.Module):
    def __init__(self, bound: float = 100.0):
        super().__init__()
        self.bound = bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + abs(x / self.bound))


class CNPEncoder(nn.Module):

    def __init__(self, **kwargs: Any) -> None:

        super().__init__()

        # Define encoder architecture
        self.layers = DenseResidualNet(
            input_dim=2,  # Wavelength and flux
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Construct encoder input: Reshape grid into batch dimension
        batch_size, n_bins, n_channels = x.shape
        x_in = x.reshape(batch_size * n_bins, n_channels)

        # Compute forward pass through network
        x_out = self.layers(x_in)
        _, latent_size = x_out.shape

        # Reshape to get grid dimension back
        x_out = x_out.reshape(batch_size, n_bins, latent_size)

        # Aggregate along wavelength dimension to get final representation
        x_out = torch.mean(x_out, dim=1)

        return torch.Tensor(x_out)


class ConvNet(nn.Module):
    """
    This implements a convolutional neural network with an architecture
    similar to the one described by Ardevol Martinez et al. (2022).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_channels: int = 1,
        **_: Any,
    ) -> None:

        super().__init__()

        # Define the layers; ignore the warning about LazyLinear
        with catch_warnings():
            filterwarnings("ignore", message="Lazy modules are a new feature")
            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=16,
                    kernel_size=17,
                    padding="same",
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=9,
                    padding="same",
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=7,
                    padding="same",
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten(),
                nn.LazyLinear(
                    out_features=128,
                ),
                nn.Linear(
                    in_features=128,
                    out_features=output_dim,
                ),
            )

        # Do one forward pass to initialize the lazy linear layer
        self.layers(torch.zeros(1, input_channels, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(self.layers(x))


class TransformerEmbedding(nn.Module):
    """
    This implements a transformer-based embedding network.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        n_heads: int,
        n_blocks: int,
    ) -> None:

        super().__init__()

        # Define "positional encodings"
        # These are the wavelengths, rescaled to the interval [0, 1], and
        # tiled along the third dimension to match the latent dimension. We
        # then apply a positional encoding to each wavelength and add it to
        # the rescaled and tiled wavelengths.
        self.positional_encoder = nn.Sequential(
            Rescale(),
            Unsqueeze(dim=2),
            Tile(shape=(1, 1, latent_dim)),
            Summer(PositionalEncoding1D(latent_dim)),
        )

        # Define the input encoder
        # We first unsqueeze the input flux along the third dimension to get
        # a shape of `(batch_size, n_bins, 1)`, and then apply a linear layer
        # with 1 input feature. This is equivalent to applying the same linear
        # layer to each wavelength separately. Then final output of the input
        # encoder has shape `(batch_size, n_bins, latent_dim)`.
        self.input_encoder = nn.Sequential(
            SoftClip(100.0),
            Unsqueeze(dim=2),
            nn.Linear(
                in_features=1,
                out_features=latent_dim,
            ),
            nn.GELU(),
        )

        # Define the transformer layers
        # These take an input of shape `(batch_size, n_bins, 2 * latent_dim)`,
        # where the last dimension contains the positional and input encodings,
        # and return an output of shape `(batch_size, latent_dim)` (computed as
        # the mean of the transformer output along the wavelength dimension)
        # that we pass through a final linear layer to get the final output.
        self.layers = nn.Sequential(
            nn.TransformerEncoder(  # type: ignore
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=n_heads,
                    dim_feedforward=2048,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=n_blocks,
            ),
            Mean(dim=1),
            nn.Linear(in_features=latent_dim, out_features=output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network.
        The shape of the input `x` is `(batch_size, n_bins, 2)`, where
        the last dimension contains the flux and the wavelength.
        """

        # Split input into flux and wavelength
        flux, wavelength = x[:, :, 0], x[:, :, 1]

        # Compute positional encoding of wavelengths
        positional_encodings = self.positional_encoder(wavelength)

        # Compute input encoding of flux
        input_embeddings = self.input_encoder(flux)

        # Add positional and input encodings (alternatively, we could also
        # concatenate them along the final dimension)
        x = input_embeddings + positional_encodings
        # x = torch.cat((positional_encodings, input_embeddings), dim=2)

        # Compute forward pass through transformer layers
        x = self.layers(x)

        return x
