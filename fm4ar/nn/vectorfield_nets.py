"""
Methods for creating vector field neural networks (for FMPE).
This is similar to embedding networks, but we have slightly different
constraints for the input and output dimensions (e.g., GLU input).
"""

from typing import Any

import torch

from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.torchutils.weights import load_and_or_freeze_model_weights


def create_vectorfield_net(
    dim_input: int,
    dim_glu: int | None,
    dim_output: int,
    network_config: dict[str, Any],
) -> torch.nn.Module:
    """
    Create a vectorfield neural network for FMPE.

    The vectorfield network receives the embedded context and the
    embedded `(t, theta)` as input and predicts the vectorfield.

    Args:
        dim_input: The input dimension of the vectorfield network.
        dim_glu: The dimension of the gated linear unit (GLU) input.
        dim_output: The output dimension of the vectorfield network.
        network_config: Configuration dictionary for the vectorfield
            network. The current implementation only supports a single
            `DenseResidualNet` block; we may extend this if we find
            that we need more flexibility.

    Returns:
        The vector field neural network.
    """

    # Get network type and keyword arguments
    network_type = network_config["network_type"]
    network_kwargs = network_config.get("kwargs", {})

    # Get parameters for freezing weights, or loading weights from a file
    freeze_weights = network_config.get("freeze_weights", False)
    load_weights = network_config.get("load_weights", {})

    # Create the vectorfield network with the given keyword arguments
    match network_type:
        case "DenseResidualNet":
            vectorfield_net = DenseResidualNet(
                input_shape=(dim_input,),
                output_dim=dim_output,
                context_features=dim_glu,
                **network_kwargs,
            )
        case _:  # pragma: no cover
            raise ValueError(f"Invalid network_type: {network_type}!")

    # Load and / or freeze weights of the vectorfield network
    load_and_or_freeze_model_weights(
        model=vectorfield_net,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    return vectorfield_net
