"""
Different embedding networks and convenience functions.
"""

from abc import ABC
from collections.abc import Mapping
from math import pi
from typing import Any

import torch
import torch.nn as nn

from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.utils.torchutils import load_and_or_freeze_model_weights


class SupportsDictInput(ABC):
    """
    Abstract base class for blocks that support dictionary inputs.
    """

    # The required keys for the input dictionary.
    # This is needed to create the dummy input that is used to determine the
    # output of the block. This is a class attribute because we might want to
    # access it without instantiating the class.
    required_keys: list[str] = ["theta", "flux", "wlen"]

    def forward(self, x: Mapping[str, torch.Tensor]) -> Any:
        """
        Forward pass through the block with a dictionary input.
        """
        raise NotImplementedError  # pragma: no cover


def block_type_string_to_class(block_type: str) -> type:
    """
    Convert a block type string to the corresponding class.

    Args:
        block_type: The string representation of the class.

    Returns:
        The class.
    """

    match block_type:
        case "Concatenate":
            return Concatenate
        case "DenseResidualNet":
            return DenseResidualNet
        case "PositionalEncoding":
            return PositionalEncoding
        case "SoftClipFlux":
            return SoftClipFlux
        case _:  # pragma: no cover
            raise ValueError(f"Invalid block type: {block_type}!")


def determine_output_shape(
    input_shape: tuple[int, ...],
    block: nn.Module,
) -> tuple[int, ...]:
    """
    Determine the output shape of a block.

    Args:
        input_shape: The shape of the input to the block (without the
            batch dimension).
        block: The block for which to determine the output shape.

    Returns:
        The shape of the output of the block.
    """

    # Define an arbitrary batch size
    batch_size = 19

    # Ignore the type of `dummy_input`
    dummy_input: torch.Tensor | dict[str, torch.Tensor]

    # Create some dummy input
    # Using `device="meta"` does not seem to work here?
    if isinstance(block, SupportsDictInput):
        dummy_input = {
            key: torch.randn((batch_size, *input_shape))
            for key in block.required_keys
        }
    else:
        dummy_input = torch.randn((batch_size, *input_shape))

    # Determine the output shape (and drop the batch dimension)
    dummy_output = block(dummy_input)
    if isinstance(dummy_output, dict):
        output_shape = tuple(dummy_output["flux"].shape[1:])
    else:
        output_shape = tuple(dummy_output.shape[1:])

    return output_shape


def create_embedding_net(
    input_shape: tuple[int, ...],
    block_configs: list[dict],
    supports_dict_input: bool = False,
) -> tuple[nn.Module, int]:
    """
    Create an embedding net from the given keyword arguments.

    Note: An embedding network can consist of multiple "blocks", where
    each block is a separate network or transformation. This function
    creates the entire embedding network from the given configuration.

    Args:
        input_shape: The shape of the input to the embedding net. For
            context embedding nets, this should be (dim_context, ).
        block_configs: A list of dictionaries with the configuration of
            the embedding net blocks. See the documentation of the
            `create_embedding_net_block()` function for more details.
        supports_dict_input: Whether the embedding net should support
            dictionary inputs. If True, the first block must be a
            `SupportsDictInput` block that can handle the context dict.

    Returns:
        A 2-tuple, `(embedding_net, embedding_dim)`, consisting of the
        embedding network and the dimension of the embedding space.
    """

    # If the embedding net is used for the context, the first block must be a
    # `SupportsDictInput` block that can handle the context dictionary.
    if supports_dict_input:
        block_type = block_type_string_to_class(block_configs[0]["block_type"])
        if not issubclass(block_type, SupportsDictInput):
            raise ValueError("The first block must be a `SupportsDictInput`!")

    # Create and collect the embedding net blocks
    embedding_net = nn.Sequential()
    for block_config in block_configs:

        # Create the stage and determine the output dimension
        block, output_shape = create_embedding_net_block(
            input_shape=input_shape,
            block_config=block_config,
        )
        embedding_net.append(block)

        # Update input_dim: The output dimension of the current stage is
        # the input dimension of the next stage
        input_shape = output_shape

    # Ensure that the final output has shape `(embedding_dim, )` when ignoring
    # the batch dimension. This is necessary because the final output of the
    # embedding net is passed to the normalizing flow or the vectorfield net,
    # which expects do not support multi-dimensional embeddings.
    # This is currently not tested because there are no blocks at the moment
    # that create multi-dimensional embeddings.
    if len(input_shape) != 1:  # pragma: no cover
        raise ValueError(
            "The final output dimension of the embedding net should be 1D, "
            f"but is {len(input_shape)}D!"
        )

    return embedding_net, input_shape[0]


def create_embedding_net_block(
    input_shape: tuple[int, ...],
    block_config: dict,
) -> tuple[nn.Module, tuple[int, ...]]:
    """
    Create an embedding net stage from the given keyword arguments.

    Args:
        input_shape: The shape of the input to the embedding net stage.
            For context embedding nets, this should be (dim_context, ).
        block_config: A dictionary with the configuration of the block.
            It must contain the following keys:
              - "model_type": The type of the block.
            Optionally, it may contain the following keys:
              - "kwargs": Additional keyword arguments for the block.
              - "freeze_weights": Whether to freeze the block's weights.
              - "load_weights": A dictionary specifying the file from
                   which to load the block's weights. For more details,
                   see `load_and_or_freeze_model_weights()`.

    Returns:
        A 2-tuple, `(block, output_shape)`, consisting of the embedding
        net block and the shape of its output.
    """

    # Get block type and keyword arguments
    block_type = block_config["block_type"]
    block_kwargs = block_config.get("kwargs", {})

    # Get parameters for freezing weights, or loading weights from a file
    freeze_weights = block_config.get("freeze_weights", False)
    load_weights = block_config.get("load_weights", {})

    # Create the block with the given keyword arguments
    # Depending on the block type, we might need to pass the input shape.
    block_class = block_type_string_to_class(block_type)
    if getattr(block_class, "requires_input_shape", False):
        block = block_class(input_shape=input_shape, **block_kwargs)
    else:
        block = block_class(**block_kwargs)

    # Load pre-trained weights and / or freeze the weights
    load_and_or_freeze_model_weights(
        model=block,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    # Determine the output shape of the block
    output_shape = determine_output_shape(input_shape=input_shape, block=block)

    return block, output_shape


class Concatenate(SupportsDictInput, nn.Module):
    """
    Concatenate the context dictionary into a single tensor.
    This can be used as the first block in a context embedding net to
    convert the context dictionary into a single tensor.
    """

    requires_input_shape = False

    def __init__(self, keys: list[str]) -> None:
        """
        Instantiate a `ConcatenateContext` block.

        Args:
            keys: The keys of the context dictionary to concatenate.
        """

        super().__init__()

        self.keys = keys
        self.required_keys = keys  # TODO: Is there a better way?

    def forward(self, context: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the `ConcatenateContext` block.
        """

        # Concatenate the context dictionary into a single tensor
        context_tensor = torch.cat([context[key] for key in self.keys], dim=1)

        return context_tensor


class PositionalEncoding(nn.Module):
    """
    A positional encoding module that can be used to encode the time
    and/or the target parameters.
    """

    requires_input_shape = False

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


class SoftClipFlux(SupportsDictInput, nn.Module):
    """
    Soft-clip normalization of the flux based on Vasist et al. (2023).
    """

    def __init__(self, bound: float = 100.0):
        super().__init__()
        self.bound = bound

    def forward(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Apply the soft clip transform to the flux.
        """

        # Create a shallow copy of the input dictionary because we do not
        # want to modify the original input in place.
        output = dict(x)

        # Clip the flux. We do not need a deep copy here first because we
        # replace the entire "flux" key in the output dictionary.
        output["flux"] = x["flux"] / (1 + torch.abs(x["flux"] / self.bound))

        return output
