"""
Methods for neural posterior estimation (NPE) models.
"""

# from functools import lru_cache
from typing import Any

import torch
from torch import nn as nn

from fm4ar.models.base import Base
from fm4ar.nn.embedding_nets import create_embedding_net
from fm4ar.nn.flows import FlowWrapper, create_flow_wrapper
# from fm4ar.utils.misc import freeze_args


class NPEModel(Base):

    # Add type hint for the network
    network: "NPENetwork"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def initialize_network(self) -> None:
        """
        Initialize the NPE network.
        """

        # Fix the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create the NPE network
        self.network = create_npe_network(model_config=self.config["model"])

    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the log probability of the given `theta`.
        """

        self.network.eval()

        context_embedding = self.network.get_context_embedding(context)
        log_prob = self.network.flow_wrapper.log_prob(
            theta=theta,
            context=context_embedding,
        )

        return log_prob

    def sample_batch(
        self,
        context: dict[str, torch.Tensor],
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the model and return the samples. If `context` is
        None, we need to specify the number of samples to draw;
        otherwise, we assume that the number of samples is the same as
        the batch size of `context`.
        """

        self.network.eval()

        context_embedding = self.network.get_context_embedding(context)
        samples = self.network.flow_wrapper.sample(
            num_samples=num_samples,
            context=context_embedding,
        )

        return samples

    def sample_and_log_prob_batch(
        self,
        context: dict[str, torch.Tensor],
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model and return the samples and their log
        probabilities. If `context` is None, we need to specify the
        number of samples to draw; otherwise, we assume that the
        number of samples is the same as the batch size of `context`.
        """

        self.network.eval()

        context_embedding = self.network.get_context_embedding(context)
        samples, log_prob = self.network.flow_wrapper.sample_and_log_prob(
            num_samples=num_samples,
            context=context_embedding,
        )

        return samples, log_prob

    def loss(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the loss for the given `theta` and `context` (i.e.,
        the mean negative log probability).
        """

        logprob = self.network(theta=theta, context=context)
        return -logprob.mean()  # type: ignore


class NPENetwork(nn.Module):
    """
    This class is a wrapper that combines an embedding net for the
    context with the actual discrete flow that models the posterior.
    """

    def __init__(
        self,
        context_embedding_net: nn.Module,
        flow_wrapper: FlowWrapper,
    ) -> None:
        """
        Initialize a NPENetwork instance.

        Args:
            context_embedding_net: The context embedding network.
            flow_wrapper: Wrapped version of the actual (discrete) flow.
        """

        super().__init__()

        self.context_embedding_net = context_embedding_net
        self.flow_wrapper = flow_wrapper

    # @freeze_args
    # @lru_cache(maxsize=1)
    def get_context_embedding(
        self,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get the embedding of the context. We wrap this in a separate
        method to allow caching the last result. Since the context is
        a dictionary, we need to convert it to a `frozendict` first to
        allow caching with the `lru_cache` decorator.
        """

        return torch.Tensor(self.context_embedding_net(context))

    def forward(
        self,
        theta: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the model. This returns the log probability
        of `theta` given the `context`, which is what we need when we
        train the model using the NPE loss function.
        """

        # Get the embedding of the context (with caching)
        context_embedding = self.get_context_embedding(context=context)

        return self.flow_wrapper.log_prob(
            theta=theta,
            context=context_embedding,
        )


def create_npe_network(model_config: dict) -> NPENetwork:
    """
    Create a `NPENetwork` instance from the given model configuration.

    Args:
        model_config: Keyword arguments specifying the model, that is,
            the "model" section of the configuration file.

    Returns:
        The NPE network wrapper.
    """

    # Extract dimensions of `theta` and `context`
    # Note: These are *not* in the config file, but are added by the
    # `prepare_new()` method that is called at the start of training.
    dim_theta = int(model_config["dim_theta"])
    dim_context = int(model_config["dim_context"])

    # Construct an embedding network for the context
    context_embedding_net, dim_embedded_context = create_embedding_net(
        input_shape=(dim_context,),
        block_configs=model_config["context_embedding_net"],
        supports_dict_input=True,
    )

    # Construct the actual discrete normalizing flow
    # We use a `FlowWrapper` to wrap the actual flow and handle the different
    # conventions used by different flows libraries (normflows vs. glasflow)
    flow_wrapper = create_flow_wrapper(
        dim_theta=dim_theta,
        dim_context=dim_embedded_context,
        flow_wrapper_config=model_config["flow_wrapper"],
    )

    # Bundle everything into a `NPENetwork` wrapper
    network = NPENetwork(
        flow_wrapper=flow_wrapper,
        context_embedding_net=context_embedding_net,
    )

    return network
