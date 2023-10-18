"""
Different embedding networks and convenience functions.
"""

from collections.abc import Sequence
from os.path import expandvars
from math import pi, log
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from fm4ar.nn.modules import Mean
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.utils.ieee754 import float2bits
from fm4ar.utils.torchutils import (
    get_mlp,
    load_and_or_freeze_model_weights,
    validate_shape,
)


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

    # Ensure that the final output has shape (embedding_dim, ). [We ignore the
    # batch dimension here.] Multi-dimensional embeddings are not supported.
    # We check against `input_dim` because this always contains the output
    # dimension of the last stage, even if the embedding net is the identity.
    if len(input_dim) != 1:
        raise ValueError(
            "The final output dimension of the embedding net "
            f"should be 1, but is {len(input_dim)}!"
        )
    else:
        final_output_dim = int(input_dim[0])

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

    # Define shortcuts
    model_type = embedding_net_stage_kwargs["model_type"]
    kwargs = embedding_net_stage_kwargs.get("kwargs", {})

    # Get parameters for freezing weights, or loading weights from a file
    freeze_weights = embedding_net_stage_kwargs.pop("freeze_weights", False)
    load_weights = embedding_net_stage_kwargs.pop("load_weights", {})

    # Create the stage
    stage: nn.Module
    match model_type:
        case "DenseResidualNet":
            if len(input_dim) != 1:
                raise ValueError("DenseResidualNet only supports 1D inputs!")
            stage = DenseResidualNet(input_dim=input_dim[0], **kwargs)
        case "DropFeatures":
            stage = DropFeatures()
        case "Float2Bits":
            stage = Float2Bits()
        case "Float2BitsTransformer":
            stage = Float2BitsTransformer(input_dim=input_dim[-1], **kwargs)
        case "PrecomputedPCAEmbedding":
            stage = PrecomputedPCAEmbedding(**kwargs)
        case "PositionalEncoding":
            stage = PositionalEncoding(**kwargs)
        case "RescaleFlux":
            stage = RescaleFlux(**kwargs)
        case "RescaleWavelength":
            stage = RescaleWavelength(**kwargs)
        case "SoftClip":
            stage = SoftClip(**kwargs)
        case "StandardizeIndividually":
            stage = StandardizeIndividually(input_dim=input_dim[0], **kwargs)
        case "SubsampleSpectrum":
            stage = SubsampleSpectrum(**kwargs)
        case "TransformerEmbedding":
            stage = TransformerEmbedding(input_dim=input_dim[-1], **kwargs)
        case _:
            raise ValueError(f"Invalid model type: {model_type}!")

    # Load pre-trained weights or freeze the weights of the flow
    load_and_or_freeze_model_weights(
        model=stage,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    # Create some dummy input to determine the output dimension of the stage.
    # Note: Using `device="meta"` seems to break the transformer embedding?
    batch_size = 19  # arbitrary
    dummy_input = torch.randn((batch_size, *input_dim))
    output_dim = stage(dummy_input).shape[1:]  # drop the batch dimension

    return stage, output_dim


class DropFeatures(nn.Module):
    """
    A module that drops some of the input features.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, n_bins, n_features)
        Output shape: (batch_size, n_bins)
        """

        # Expected shape: (batch_size, n_bins, n_features)
        validate_shape(x, (None, None, None))
        batch_size, n_bins, n_features = x.shape

        # Only keep the first feature (flux)
        x = x[:, :, 0]
        validate_shape(x, (batch_size, n_bins))

        return x


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
    """
    Soft-clip normalization of the flux based on Vasist et al. (2023).
    """

    def __init__(self, bound: float = 100.0):
        super().__init__()
        self.bound = bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, n_bins, n_features)
        Output shape: (batch_size, n_bins, n_features)
        """

        validate_shape(x, (None, None, None))

        # Split input features
        flux, *other_features = torch.split(x, 1, dim=2)

        # Apply the soft clip (only to the flux)
        flux = flux / (1 + torch.abs(flux / self.bound))

        # Reassemble input features
        x = torch.stack((flux, *other_features), dim=2).squeeze(3)

        validate_shape(x, (None, None, None))

        return x


class TransformerEmbedding(nn.Module):
    """
    This implements a transformer-based embedding network.
    """

    def __init__(
        self,
        input_dim: int,  # number of input features (flux, wavelength, noise)
        latent_dim: int,  # dimension of latent embeddings
        output_dim: int,   # dimension of output embeddings
        n_heads: int,
        n_blocks: int,
    ) -> None:
        """
        Instantiate a TransformerEmbedding module.
        """

        super().__init__()

        # Store the parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Define layers
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=latent_dim // 2),
            nn.Tanh(),
        )
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(  # type: ignore
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=n_heads,
                    dim_feedforward=1024,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=n_blocks,
            ),
            Mean(dim=1),
            nn.Linear(in_features=latent_dim, out_features=output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network.
        """

        # Expected shape: (batch_size, n_bins, n_features)
        validate_shape(x, (None, None, self.input_dim))
        batch_size, n_bins, _ = x.shape

        # Split input features
        flux, wlen, *_ = torch.split(x, 1, dim=2)
        wlen = wlen.squeeze(2)

        if self.latent_dim % 4 != 0:
            raise ValueError("The latent dimension must be even!")

        # Compute positional encoding for the wavelengths
        freqs = []
        for i in range(int(self.latent_dim / 4)):
            freqs.append(torch.sin(wlen / 10_000 ** (4 * i / self.latent_dim)))
            freqs.append(torch.cos(wlen / 10_000 ** (4 * i / self.latent_dim)))
        encoded_wlen = torch.stack(freqs, dim=2)
        validate_shape(
            encoded_wlen, (batch_size, n_bins, self.latent_dim // 2)
        )

        # Send the flux through the MLP
        encoded_flux = self.mlp(flux)
        validate_shape(
            encoded_flux, (batch_size, n_bins, self.latent_dim // 2)
        )

        # Combine the flux and the positional encoding
        transformer_input = torch.cat([encoded_flux, encoded_wlen], dim=2)

        # Apply the embedding network
        # Expected shape: (batch_size, output_dim)
        output = self.transformer(transformer_input)
        validate_shape(output, (batch_size, self.output_dim))

        return torch.Tensor(output)


class RescaleWavelength(nn.Module):
    """
    Pre-processing module that rescales the wavelengths to the interval
    [-1, 1]. This should be used before any proper embedding module.
    """

    def __init__(self, min_wavelength: float, max_wavelength: float) -> None:
        super().__init__()
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, n_bins, n_features)
        Output shape: (batch_size, n_bins, n_features)
        """

        # Expected shape: (batch_size, n_bins, n_features)
        validate_shape(x, (None, None, None))
        batch_size, n_bins, n_features = x.shape

        # Split input features
        flux, wavelengths, *other_features = torch.split(x, 1, dim=2)

        # Rescale wavelengths
        wavelengths = 2 * (
            (wavelengths - self.min_wavelength)
            / (self.max_wavelength - self.min_wavelength)
            - 0.5
        )

        # Reassemble input features
        x = torch.cat((flux, wavelengths, *other_features), dim=2)
        validate_shape(x, (batch_size, n_bins, n_features))

        return x


class RescaleFlux(nn.Module):
    """
    Pre-processing module to standardize in particular the spectra of
    the Vasist-2023 dataset, which cover many orders of magnitude.
    """

    def __init__(self, mode: str, *_: Any, **__: Any) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Expected shape: (batch_size, n_bins, n_features)
        validate_shape(x, (None, None, None))
        batch_size, n_bins, n_features = x.shape

        # Split input features
        flux, *other_features = torch.split(x, 1, dim=2)

        if self.mode == "log":
            flux = torch.log10(1 + flux)  # Flux may be 0 in some bins
            x = torch.cat((flux, *other_features), dim=2)
            validate_shape(x, (batch_size, n_bins, n_features))
            return x

        if self.mode == "tuple":

            # Take mean and std along the wavelength dimension (over all bins)
            mean = torch.mean(flux, dim=1, keepdim=True).repeat(1, n_bins, 1)
            std = torch.std(flux, dim=1, keepdim=True).repeat(1, n_bins, 1)

            # Standardize the flux. We use log(1 + ...) to prevent that we get
            # NaNs if the mean or standard deviation are ever zero. [This has
            # happened for the Vasist-2023 dataset.]
            x1 = (flux - mean) / std
            x2 = torch.log10(1 + mean)
            x3 = torch.log10(1 + std)

            # Reassemble input features. We add the mean and std of the flux
            # at the end to keep the interpretation the same as before (i.e.,
            # x[0] is the flux, x[1] the wavelengths, and x[2] the uncertainty)
            x = torch.cat([x1, *other_features, x2, x3], dim=2)
            validate_shape(x, (batch_size, n_bins, n_features + 2))
            return x

        raise ValueError(f"Invalid mode: {self.mode}!")


class SubsampleSpectrum(nn.Module):

    # TODO: Should this functionality be moved to the data loader entirely?

    def __init__(self, min_fraction: float = 0.1):
        super().__init__()
        self.min_fraction = min_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, n_bins_original, n_features)
        Output shape: (batch_size, n_bins_subsampled, n_features)
        """

        # Expected shape: (batch_size, n_bins_original, n_features)
        validate_shape(x, (None, None, None))
        batch_size, n_bins, n_features = x.shape

        if self.training:
            fraction = float(np.random.uniform(self.min_fraction, 1.0))
            n_bins = x.shape[1]
            idx = torch.randperm(n_bins)
            mask = idx < (fraction * n_bins)
            x = x[:, mask, :]

        # Expected shape: (batch_size, n_bins_resampled, n_features)
        validate_shape(x, (batch_size, None, n_features))

        return x


class Float2Bits(nn.Module):
    """
    Standardize the input data by casting them to a bit tensor
    """

    def __init__(self) -> None:

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=32, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=64),
            nn.GELU(),
            nn.Linear(in_features=64, out_features=16),
            nn.GELU(),
            nn.Linear(in_features=16, out_features=4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        validate_shape(x, (None, None))
        batch_size, n_bins = x.shape

        # Make sure there are no subnormal numbers in x
        x[torch.abs(x) < 2e-38] = 0.0

        # Cast to bit tensor
        x = float2bits(x, precision="single")
        validate_shape(x, (batch_size, n_bins, 32))

        # Send the bit tensor through the MLP
        x = self.mlp(x)
        validate_shape(x, (batch_size, n_bins, 4))

        # Flatten out the last dimension
        x = x.view(batch_size, n_bins * 4)
        validate_shape(x, (batch_size, n_bins * 4))

        return x


class StandardizeIndividually(nn.Module):
    """
    Standardize each spectrum individually and then add the mean and
    standard deviation either via GLU or via concatenation.
    """

    def __init__(
        self,
        input_dim: int,
        mode: Literal["glu", "concat"],
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.mode = mode

        # Define small networks for the mean and std
        self.mean_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=input_dim),
        )
        self.std_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=input_dim),
        )

        # Define GLU
        self.glu = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Check the input shape
        validate_shape(x, (None, None))
        batch_size, n_bins = x.shape

        # Standardize each spectrum individually
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = (x - mean.repeat(1, n_bins)) / std.repeat(1, n_bins)

        # Process the mean and std
        mean = torch.log10(1 + mean)
        mean = self.mean_net(mean)
        std = torch.log10(1 + std)
        std = self.std_net(std)

        # Reassemble input features depending on the mode
        if self.mode == "glu":
            x_1 = self.glu(torch.cat((x, mean), dim=1))
            x_2 = self.glu(torch.cat((x, std), dim=1))
            x = torch.cat((x_1, x_2), dim=1)
            validate_shape(x, (batch_size, 2 * n_bins))
        elif self.mode == "concat":
            x = torch.cat((x, mean, std), dim=1)
            validate_shape(x, (batch_size, n_bins * 3))
        else:
            raise ValueError(f"Invalid mode: {self.mode}!")

        return x


class Float2BitsTransformer(nn.Module):
    """
    Network that combines a float2bit standardization with a transformer
    architecture to learn context embeddings.
    """

    def __init__(
        self,
        input_dim: int,  # number of input features (flux, wavelength, noise)
        latent_dim: int,  # dimension of latent embeddings
        output_dim: int,   # dimension of output embeddings
        n_heads: int,
        n_blocks: int,
    ) -> None:
        """
        Instantiate a TransformerEmbedding module.
        """

        super().__init__()

        # Store the parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Define MLP to map the flux from bit pattern to the latent dimension
        self.flux_mlp = get_mlp(
            input_dim=32,
            output_dim=latent_dim,
            hidden_dims=(1024, 1024, 1024),
            activation="gelu",
            batch_norm=True,
        )

        # Define the transformer backbone
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(  # type: ignore
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=n_heads,
                    dim_feedforward=1024,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=n_blocks,
            ),
            Mean(dim=1),
            nn.Linear(in_features=latent_dim, out_features=output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network.
        """

        # Expected shape: (batch_size, n_bins, n_features)
        validate_shape(x, (None, None, self.input_dim))
        batch_size, n_bins, _ = x.shape

        # Split input features
        flux, wlen, *_ = torch.split(x, 1, dim=2)
        wlen = wlen.squeeze(2)

        # Compute positional encoding for the wavelengths
        encoded_wlen = self.positional_encoding(wlen, self.latent_dim)
        validate_shape(encoded_wlen, (batch_size, n_bins, self.latent_dim))

        # Map flux to bit pattern and then to latent dimension
        encoded_flux = float2bits(flux, precision="single")
        validate_shape(encoded_flux, (batch_size, n_bins, 32))
        encoded_flux = self.flux_mlp(flux)
        validate_shape(encoded_flux, (batch_size, n_bins, self.latent_dim))

        # Combine the flux and the positional encoding
        transformer_input = encoded_flux + encoded_wlen

        # Apply the embedding network
        output = self.transformer(transformer_input)
        validate_shape(output, (batch_size, self.output_dim))

        return torch.Tensor(output)

    @staticmethod
    def positional_encoding(
        x: torch.Tensor,
        latent_dim: int,
    ) -> torch.Tensor:
        """
        Compute a value-based positional encoding for the input tensor.

        Args:
            x: Tensor of shape `(batch_size, n_bins)`.
            latent_dim: The latent dimensionality for the encoding.

        Returns:
             A tensor of shape `(batch_size, n_bins, latent_dim)`.
        """

        # Ensure latent_dim is even
        if latent_dim % 2 != 0:
            raise ValueError("Latent dimension must be even.")

        # Expand dims for broadcasting
        x_exp = x.unsqueeze(-1)

        # Calculate div_term and expand dims for broadcasting
        # 1. Define the base of the exponent in the original formula
        # 2. Calculate the exponent term, which is log(base) / latent_dim
        # 3. Generate a sequence [0, 2, 4, ..., latent_dim - 2]
        # 4. Compute div_term using exp function for the decreasing series
        # 5. Expand dims for broadcasting
        base = 1 / 10_000
        exponent_term = log(base) / latent_dim
        sequence = torch.arange(0, latent_dim, 2, device=x.device).float()
        div_term = torch.exp(sequence * exponent_term)
        div_term = div_term.unsqueeze(0).unsqueeze(0)

        # Calculate sine and cosine values
        sine_terms = torch.sin(x_exp * div_term)
        cosine_terms = torch.cos(x_exp * div_term)

        # Interleave sine and cosine terms
        result = torch.stack((sine_terms, cosine_terms), dim=-1)
        result = result.view(x.shape[0], x.shape[1], latent_dim)

        return result
