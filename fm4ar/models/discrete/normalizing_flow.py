"""
Define ``NormalizingFlow`` posterior model.
"""

from typing import Any

import torch

from fm4ar.models.base import Base
from fm4ar.models.discrete.model import create_df_model, DiscreteFlowModel


class NormalizingFlow(Base):

    network: DiscreteFlowModel

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def initialize_network(self) -> None:
        self.network = create_df_model(model_kwargs=self.config["model"])

    def log_prob_batch(
        self,
        y: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.Tensor(self.network(y, *context_data))

    def sample_batch(
        self,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.Tensor(self.network.sample(*context_data))

    def sample_and_log_prob_batch(
        self,
        *context_data: torch.Tensor,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples, log_probs = self.network.sample_and_log_prob(*context_data)
        return torch.Tensor(samples), torch.Tensor(log_probs)

    def loss(
        self,
        data: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.Tensor(-self.network(data, *context_data).mean())
