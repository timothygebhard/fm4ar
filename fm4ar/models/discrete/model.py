"""
Wrapper class and helper functions for discrete flow models.
"""

from typing import overload

import torch
import torch.nn as nn

from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.nn.flows import (
    FlowWrapper,
    create_glasflow_flow,
    create_normflows_flow,
)
from fm4ar.utils.torchutils import load_and_or_freeze_model_weights


class DiscreteFlowModel(nn.Module):
    """
    This class is a wrapper that combines an embedding net for the
    context with the actual discrete flow that models the posterior.
    """

    def __init__(
        self,
        flow_wrapper: FlowWrapper,
        context_embedding_net: nn.Module,
    ) -> None:
        """
        Initialize a DiscreteFlowModel instance.

        Args:
            flow_wrapper: Wrapped version of the actual (discrete) flow
                that models the posterior.
            context_embedding_net: The context embedding network.
        """

        super().__init__()

        self.context_embedding_net = context_embedding_net
        self.flow_wrapper = flow_wrapper

    @overload
    def get_context_embedding(self, context: None) -> None:
        ...

    @overload
    def get_context_embedding(self, context: torch.Tensor) -> torch.Tensor:
        ...

    # TODO: Should we bring back caching here?
    def get_context_embedding(
        self,
        context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Get the embedding of the context.
        """

        if context is None:
            return None
        else:
            return torch.Tensor(self.context_embedding_net(context))

    def forward(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass through the model. This returns the log probability
        of `theta` given the `context`, which is what we need when we
        train the model using the NPE loss function.
        """

        return self.flow_wrapper.log_prob(theta=theta, context=context)


def create_df_model(model_kwargs: dict) -> DiscreteFlowModel:
    """
    Create a discrete flow model from the given kwargs, which consists
    of an embedding network for the context and a discrete flow network.

    Args:
        model_kwargs: Keyword arguments specifying the model. (The
            "model" section of the configuration file.)

    Returns:
        The discrete flow model.
    """

    # Extract dimensions of `theta` and `context`.
    # For the context, we allow a tuple of dimensions to handle cases where
    # the inputs are more than 1-dimensional (e.g., multi-channel spectra).
    theta_dim = int(model_kwargs["theta_dim"])
    context_dim = (
        (model_kwargs["context_dim"],)
        if isinstance(model_kwargs["context_dim"], int)
        else tuple(model_kwargs["context_dim"])
    )

    # Construct an embedding network for the context
    context_embedding_kwargs = model_kwargs.get("context_embedding_kwargs", {})
    context_embedding_net, embedded_context_dim = create_embedding_net(
        input_dim=context_dim,
        embedding_net_kwargs=context_embedding_kwargs,
    )

    # Define some shortcuts
    posterior_kwargs = model_kwargs["posterior_kwargs"]
    flow_library = posterior_kwargs.pop("flow_library", "glasflow")
    freeze_weights = posterior_kwargs.pop("freeze_weights", False)
    load_weights = posterior_kwargs.pop("load_weights", {})

    # Construct the actual discrete normalizing flow
    if flow_library == "glasflow":
        flow_wrapper = create_glasflow_flow(
            theta_dim=theta_dim,
            context_dim=embedded_context_dim,
            posterior_kwargs=posterior_kwargs,
        )
    elif flow_library == "normflows":
        flow_wrapper = create_normflows_flow(
            theta_dim=theta_dim,
            context_dim=embedded_context_dim,
            posterior_kwargs=posterior_kwargs,
        )
    else:
        raise ValueError(f"Unknown flow library: {flow_library}")

    # Load pre-trained weights or freeze the weights of the flow
    load_and_or_freeze_model_weights(
        model=flow_wrapper.flow,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    # Combine the flow and the context embedding net
    df_model = DiscreteFlowModel(
        flow_wrapper=flow_wrapper,
        context_embedding_net=context_embedding_net,
    )

    return df_model
