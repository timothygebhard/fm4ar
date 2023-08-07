"""
Wrapper class and helper functions for continuous flow models.
"""

from functools import lru_cache

import torch
import torch.nn as nn

from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.nn.resnets import DenseResidualNet
from fm4ar.utils.torchutils import (
    forward_pass_with_unpacked_tuple,
    validate_dims,
)


class ContinuousFlowModel(nn.Module):
    """
    This class is a wrapper that combines an embedding net for both
    the context and (t, theta) with the actual continuous flow, that
    is, the network that learns the vector field.
    """

    def __init__(
        self,
        vectorfield_net: nn.Module,
        context_embedding_net: nn.Module,
        t_theta_embedding_net: nn.Module,
        context_with_glu: bool,
        t_theta_with_glu: bool,
    ) -> None:
        """
        Instantiate a new continuous flow model.

        Args:
            vectorfield_net: The network that learns the vector field.
            context_embedding_net: The context embedding network.
            t_theta_embedding_net: The (t, theta) embedding network.
            context_with_glu: Whether to use a gated linear unit (GLU)
                for the context embedding.
            t_theta_with_glu: Whether to use a gated linear unit (GLU)
                for the (t, theta) embedding.
        """

        super(ContinuousFlowModel, self).__init__()

        self.vectorfield_net = vectorfield_net
        self.context_embedding_net = context_embedding_net
        self.t_theta_embedding_net = t_theta_embedding_net
        self.context_with_glu = context_with_glu
        self.t_theta_with_glu = t_theta_with_glu

    @lru_cache(maxsize=1)
    def get_context_embedding(self, *context: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding of the context. We wrap this in a separate
        method to allow caching the last result.
        """

        return forward_pass_with_unpacked_tuple(
            self.context_embedding_net,
            *context,
        )

    def forward(
        self,
        t: torch.Tensor,
        theta: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the continuous flow model, that is,
        compute the embeddings of the context and (t, theta) and
        predict the vector field.
        """

        # Concatenate `t` and `theta` and embed the result
        t_theta = torch.cat((t.unsqueeze(1), theta), dim=1)
        t_theta_embedding = self.t_theta_embedding_net(t_theta)
        validate_dims(t_theta_embedding, 2)

        # Handle unconditional forward pass
        if len(context) == 0:
            return torch.Tensor(self.vectorfield_net(t_theta_embedding))

        # Get the embedding of the context
        context_embedding = self.get_context_embedding(*context)
        validate_dims(context_embedding, 2)

        # Collect inputs for the continuous flow network:
        # There are two entry points, one "normal" and one via a GLU.
        if self.context_with_glu and self.t_theta_with_glu:
            cf_input = torch.empty((len(theta), 0), device=theta.device)
            glu_context = torch.cat((context_embedding, t_theta_embedding), 1)
        elif not self.context_with_glu and not self.t_theta_with_glu:
            cf_input = torch.cat((context_embedding, t_theta_embedding), 1)
            glu_context = None
        elif self.context_with_glu and not self.t_theta_with_glu:
            cf_input = t_theta_embedding
            glu_context = context_embedding
        elif not self.context_with_glu and self.t_theta_with_glu:
            cf_input = context_embedding
            glu_context = t_theta_embedding
        else:
            raise RuntimeError("This should never happen!")

        if glu_context is None:
            return torch.Tensor(self.vectorfield_net(cf_input))
        return torch.Tensor(self.vectorfield_net(cf_input, glu_context))


def create_cf_model(model_kwargs: dict) -> ContinuousFlowModel:
    """
    Create a continuous flow model from the given kwargs, which consists
    of an embedding network for the context, an embedding network for
    theta, and and a continuous flow network.

    Args:
        model_kwargs: Keyword arguments specifying the model. (The
            "model" section of the configuration file.)

    Returns:
        The continuous flow model.
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

    # Check if we use GLU for embedded `t_theta` and / or `context`
    t_theta_with_glu = model_kwargs.get("t_theta_with_glu", False)
    context_with_glu = model_kwargs.get("context_with_glu", False)

    # Sanity check: We can only use GLU if we use a DenseResidualNet for the
    # continuous flow network (usually, this is only relevant if we want to
    # train an unconditional model with a "simple" continuous flow network).
    if (
        model_kwargs["posterior_kwargs"]["model_type"] != "DenseResidualNet"
        and (t_theta_with_glu or context_with_glu)
    ):
        raise ValueError(
            "Can only use GLU if `posterior_kwargs.model_type` is "
            "`DenseResidualNet`!"
        )

    # Construct an embedding network for the context
    context_embedding_kwargs = model_kwargs.get("context_embedding_kwargs", {})
    context_embedding_net, embedded_context_dim = create_embedding_net(
        input_dim=context_dim,
        embedding_net_kwargs=context_embedding_kwargs,
    )

    # Construct an embedding network for `(t, theta)`
    t_theta_embedding_kwargs = model_kwargs.get("t_theta_embedding_kwargs", {})
    t_theta_embedding_net, embedded_t_theta_dim = create_embedding_net(
        input_dim=(theta_dim + 1, ),
        embedding_net_kwargs=t_theta_embedding_kwargs,
    )

    # Compute GLU dimensions and input dimension for continuous flow network
    glu_dim = (
        t_theta_with_glu * embedded_t_theta_dim
        + context_with_glu * embedded_context_dim
    )
    input_dim = embedded_t_theta_dim + embedded_context_dim - glu_dim
    glu_dim = glu_dim if glu_dim > 0 else None

    # Construct neural network that predicts the vectorfield
    match pm_type := model_kwargs["posterior_kwargs"]["model_type"]:
        case "DenseResidualNet":
            vectorfield_net = DenseResidualNet(
                input_dim=input_dim,
                output_dim=theta_dim,
                context_features=glu_dim,
                **model_kwargs["posterior_kwargs"]["kwargs"],
            )
        case _:
            raise ValueError(f"Invalid model type: {pm_type}!")

    # Combine embedding networks and continuous flow network to a model
    model = ContinuousFlowModel(
        vectorfield_net=vectorfield_net,
        context_embedding_net=context_embedding_net,
        t_theta_embedding_net=t_theta_embedding_net,
        t_theta_with_glu=t_theta_with_glu,
        context_with_glu=context_with_glu,
    )
    return model
