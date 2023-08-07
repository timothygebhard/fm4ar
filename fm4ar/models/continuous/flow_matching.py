"""
Define `FlowMatching` model.
This is one particular instance of a continuous normalizing flow model;
in princinple, there could be others (like score-matching).
"""

from typing import Any

import torch
from torch import nn

from fm4ar.models.continuous.base import ContinuousFlowBase


class FlowMatching(ContinuousFlowBase):
    """
    Class for continuous normalizing flows trained with flow matching.

        t         ~ U[0, 1-eps)                         (noise level)
        theta_0   ~ N(0, 1)                             (sampled noise)
        theta_1   = theta                               (pure sample)
        theta_t   = c1(t) * theta_1 + c0(t) * theta_0   (noisy sample)

        eps       = 0
        c0        = (1 - (1 - sigma_min) * t)
        c1        = t

        v_target  = theta_1 - (1 - sigma_min) * theta_0
        loss      = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)

        self.eps = 0
        self.sigma_min = self.config["model"]["sigma_min"]

    def evaluate_vectorfield(
        self,
        t: float | torch.Tensor,
        theta_t: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the vectorfield at the given time and parameter values.

        Args:
            t: The time at which to evaluate the vectorfield. Usually
                a float between 0 and 1; for type consistency, we also
                allow a tensor.
            theta_t: The parameter values at which to evaluate the
                vectorfield.
            *context_data: The context (i.e., the observed data) that
                is used to condition the network.

        Returns:
            The vectorfield evaluated at the given time and parameter.
        """

        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        return torch.Tensor(self.network(t, theta_t, *context_data))

    def loss(
        self,
        theta: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates loss as the the mean squared error between the
        predicted vectorfield and the vector field for transporting the
        parameter data to samples from the prior.

        Args:
            theta: Parameters.
            *context_data: Context (i.e., observed data).

        Returns:
            Loss tensor.
        """

        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)

        true_vf = theta - (1 - self.sigma_min) * theta_0
        pred_vf = self.network(t, theta_t, *context_data)

        loss = mse(pred_vf, true_vf)

        return torch.Tensor(loss)


def ot_conditional_flow(
    theta_0: torch.Tensor,
    theta_1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """
    Optimal transport for the conditional flow.

    This function basically interpolates between theta_0 and theta_1.

    Args:
        theta_0: Parameters at t=0.
        theta_1: Parameters at t=1.
        t: Time at which to evaluate the flow.
        sigma_min: Standard deviation of the target distribution.

    Returns:
        The parameters at time t.
    """
    return torch.Tensor(
        (1 - (1 - sigma_min) * t)[:, None] * theta_0 + t[:, None] * theta_1
    )
