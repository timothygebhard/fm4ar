"""
Wrapper class and helper functions for discrete flow models.
"""

from functools import lru_cache

import torch
import torch.nn as nn
from glasflow.nflows import distributions, flows

from fm4ar.utils.torchutils import forward_pass_with_unpacked_tuple
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.nn.nsf import create_transform


class DiscreteFlowModel(nn.Module):
    """
    This class is a wrapper that combines an embedding net for the
    context with the actual discrete flow that models the posterior.

    Besides consistency with the continuous flow model, this wrapper
    exists for two reasons:
    (1) some embedding networks take tuples as input, which is not
        supported by the nflows package. This is why we handle the
        context embedding separately here.
    (2) parallelization across multiple GPUs requires a `forward()`
        method, but the relevant method for training is `log_prob()`.
    """

    def __init__(
        self,
        flow: flows.base.Flow,
        context_embedding_net: nn.Module,
    ) -> None:
        """
        Initialize a DiscreteFlowModel instance.

        Args:
            flow: The (discrete) flow object that models the posterior.
                Technically, this already supports an embedding net for
                the context, but we handle this separately here and
                assume that flow.embedding_net is the identity.
            context_embedding_net: The context embedding network.
        """

        super().__init__()

        self.context_embedding_net = context_embedding_net
        self.flow = flow

    @lru_cache(maxsize=1)
    def get_context_embedding(self, *context: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding of the context.
        """

        return forward_pass_with_unpacked_tuple(
            self.context_embedding_net,
            *context,
        )

    def log_prob(
        self,
        y: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:

        if len(context) == 0:
            return torch.Tensor(self.flow.log_prob(y))

        context_embedding = self.get_context_embedding(*context)
        return torch.squeeze(self.flow.log_prob(y, context_embedding))

    def sample(
        self,
        *context: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:

        if len(context) == 0:
            return torch.squeeze(self.flow.sample(num_samples))

        context_embedding = self.get_context_embedding(*context)
        return torch.squeeze(
            self.flow.sample(num_samples, context_embedding)
        )

    def sample_and_log_prob(
        self,
        *context: torch.Tensor,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if len(context) == 0:
            sample, log_prob = self.flow.sample_and_log_prob(num_samples)

        else:
            context_embedding = self.get_context_embedding(*context)
            sample, log_prob = self.flow.sample(num_samples, context_embedding)

        return torch.squeeze(sample), torch.squeeze(log_prob)

    def forward(
        self,
        theta: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model. This returns the log probability
        of `theta` given the `context`, which is what we need when we
        train the model using the NPE loss function.
        """

        return (
            self.log_prob(theta, *context)
            if len(context) > 0
            else self.log_prob(theta)
        )


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

    # Construct the actual discrete normalizing flow
    # We set the embedding net to the identity, because we handle the context
    # embedding separately in the `DiscreteFlowModel` wrapper.
    distribution = distributions.StandardNormal((theta_dim,))
    transform = create_transform(
        theta_dim=theta_dim,
        context_dim=embedded_context_dim,
        **model_kwargs["posterior_kwargs"],
    )
    flow = flows.Flow(
        transform=transform,
        distribution=distribution,
        embedding_net=nn.Identity(),
    )

    return DiscreteFlowModel(
        flow=flow,
        context_embedding_net=context_embedding_net,
    )
