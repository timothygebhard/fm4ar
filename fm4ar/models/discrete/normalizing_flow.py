"""
Define ``NormalizingFlow`` posterior model.
"""

from typing import Any

import torch

from fm4ar.models.base import Base
from fm4ar.models.discrete.model import create_df_model, DiscreteFlowModel


class NormalizingFlow(Base):

    model: DiscreteFlowModel

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a new NormalizingFlow instance.
        """

        super().__init__(**kwargs)

    def initialize_model(self) -> None:
        """
        Initialize the model (i.e., the discrete normalizing flow).
        """

        self.model = create_df_model(model_kwargs=self.config["model"])

    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute the log probability of the given `theta`.
        """

        log_prob = self.model.log_prob(theta, context)
        return torch.Tensor(log_prob)

    def sample_batch(
        self,
        context: torch.Tensor | None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the model and return the samples. If `context` is
        None, we need to specify the number of samples to draw;
        otherwise, we assume that the number of samples is the same as
        the batch size of `context`.
        """

        samples = self.model.sample(
            context=context,
            num_samples=num_samples,
        )
        return torch.Tensor(samples)

    def sample_and_log_prob_batch(
        self,
        context: torch.Tensor | None,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model and return the samples and their log
        probabilities. If `context` is None, we need to specify the
        number of samples to draw; otherwise, we assume that the
        number of samples is the same as the batch size of `context`.
        """

        samples, log_probs = self.model.sample_and_log_prob(
            context=context,
            num_samples=num_samples,
        )
        return torch.Tensor(samples), torch.Tensor(log_probs)

    def loss(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute the loss for the given `theta` and `context` (i.e.,
        the mean negative log probability).
        """

        return torch.Tensor(-self.model(theta=theta, context=context).mean())
