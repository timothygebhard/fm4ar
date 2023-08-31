"""
Define an abstract base class for continuous normalizing flows (CNF).
This extends the normal `Base` class by adding a method to evaluate the
vectorfield `v(theta_t, t)` that generates the flow.
"""

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torchdiffeq import odeint

from fm4ar.models.continuous.model import create_cf_model
from fm4ar.models.base import Base


class ContinuousFlowBase(Base):
    """
    Base class for continuous normalizing flows (CNF).

    CNFs are parameterized with a vector field `v(theta_t, t)`, that
    transports a simple base distribution (typically a Gaussian N(0,1)
    with same dimension as `theta`) at time `t=0` to the target
    distribution at time `t=1`.

    This vector field defines the flow via the ODE

        d/dt f(theta, t) = v(f(theta, t), t).

    The vectorfield `v` is parameterized with a neural network. It is
    impractical to train this neural network (and thereby the CNF)
    directly with log-likelihood maximization, as solving the full ODE
    for each training iteration requires thousands of vectorfield
    evaluations.

    Several alternative methods have been developed to make training
    CNFs more efficient. These directly regress on the vectorfield `v`
    (or a scaled version of `v`, such as the score). It has been shown
    that this can be done on a per-sample basis by adding noise to the
    parameters at various scales `t`. Specifically, a parameter sample
    `theta` is transformed as follows.

        t         ~ U[0, 1-eps)                         (noise level)
        theta_0   ~ N(0, 1)                             (sampled noise)
        theta_1   = theta                               (pure sample)
        theta_t   = c1(t) * theta_1 + c0(t) * theta_0   (noisy sample)

    Within that framework, one can employ different methods to learn the
    vectorfield `v`, such as flow matching or score matching. These have
    slightly different coefficients `c1(t)`, `c2(t)` and training
    objectives.
    """

    def __init__(self, **kwargs: Any) -> None:

        super().__init__(**kwargs)

        self.eps = 0
        self.time_prior_exponent = self.config["model"].get(
            "time_prior_exponent", 0
        )
        self.theta_dim = self.config["model"]["theta_dim"]

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample time `t` from a power law distribution.
        (For `time_prior_exponent=0`, this is a uniform distribution.)
        """
        t = (1 - self.eps) * torch.rand(batch_size, device=self.device)
        return torch.pow(t, 1 / (1 + self.time_prior_exponent))

    def sample_theta_0(self, batch_size: int) -> torch.Tensor:
        """
        Sample `theta_0` from a Gaussian prior.
        """
        return torch.randn(batch_size, self.theta_dim, device=self.device)

    @abstractmethod
    def evaluate_vectorfield(
        self,
        t: float,
        theta_t: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the vectorfield `v(t, theta_t, context_data)` that
        generates the flow via the ODE:

            d/dt f(theta_t, t, context)
            = v(f(theta_t, t, context), t, context).

        Args:
            t: Time (noise level).
            theta_t: Noisy parameters, perturbed with noise level `t`.
            *context_data: List with context data.
        """
        raise NotImplementedError()

    def rhs_of_joint_ode(
        self,
        t: float,
        theta_and_div_t: torch.Tensor,
        *context_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the right hand side of the neural ODE that is used to
        evaluate the log_prob of theta samples. This is a joint ODE over
        the vectorfield and the divergence. By integrating this ODE, one
        can simultaneously trace the parameter sample theta_t and
        integrate the divergence contribution to the log_prob, see e.g.,
        https://arxiv.org/abs/1806.07366 or Appendix C in
        https://arxiv.org/abs/2210.02747.

        Parameters
        ----------
        t: float
            time (noise level)
        theta_and_div_t: torch.tensor
            concatenated tensor of (theta_t, div).
            theta_t: noisy parameters, perturbed with noise level t
        *context_data: list[torch.tensor]
            list with context data (GW data)

        Returns
        -------
        torch.tensor
            vector field that generates the flow and its divergence
            (required for likelihood evaluation).
        """
        theta_t = theta_and_div_t[:, :-1]  # extract theta_t
        with torch.enable_grad():
            theta_t.requires_grad_(True)
            vf = self.evaluate_vectorfield(t, theta_t, *context_data)
            div_vf = compute_divergence(vf, theta_t)
        return torch.cat((vf, -div_vf), dim=1)

    def initialize_network(self) -> None:
        """
        Initialize the neural net that parameterizes the vectorfield.
        """
        self.network = create_cf_model(model_kwargs=self.config["model"])

    def sample_batch(
        self,
        *context_data: torch.Tensor,
        batch_size: int | None = None,
        tolerance: float = 1e-7,
    ) -> torch.Tensor:
        """
        Returns (conditional) samples for a batch of contexts by solving
        an ODE forwards in time.

        Args:
            *context_data: Context data (e.g., observed data).
            batch_size: Batch size for sampling.
                If len(context_data) > 0, we automatically determine the
                batch size from the context data, so this option is only
                used for unconditional sampling.
            tolerance: Tolerance (atol and rtol) for the ODE solver.

        Returns:
            The generated samples.
        """

        self.network.eval()

        # Ensure we got a valid combination of context_data and batch_size
        if len(context_data) == 0 and batch_size is None:
            raise ValueError("Must set batch_size for unconditional sampling!")
        if len(context_data) > 0 and batch_size is not None:
            raise ValueError("Can't set batch_size for conditional sampling!")

        # Extract batch size from context data if not set
        batch_size = len(context_data[0]) if batch_size is None else batch_size

        # Solve ODE forwards in time to get theta_1 from theta_0
        with torch.no_grad():
            theta_0 = self.sample_theta_0(batch_size)
            _, theta_1 = odeint(
                func=lambda t, theta_t: self.evaluate_vectorfield(
                    t, theta_t, *context_data
                ),
                y0=theta_0,
                t=self.integration_range,
                atol=tolerance,
                rtol=tolerance,
                method="dopri5",
            )

        return torch.Tensor(theta_1)

    def log_prob_batch(
        self,
        theta: torch.Tensor,
        *context_data: torch.Tensor,
        tolerance: float = 1e-7,
    ) -> torch.Tensor:
        """
        Evaluates log_probs of theta conditional on provided context.
        For this we solve an ODE backwards in time until we reach the
        initial pure noise distribution.

        There are two contributions, the log_prob of theta_0 (which is
        uniquely determined by theta) under the base distribution, and
        the integrated divergence of the vectorfield.

        Parameters.
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------

        """
        self.network.eval()

        div_init = torch.zeros(
            (theta.shape[0],), device=theta.device
        ).unsqueeze(1)
        theta_and_div_init = torch.cat((theta, div_init), dim=1)

        _, theta_and_div_0 = odeint(
            lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, *context_data
            ),
            theta_and_div_init,
            torch.flip(
                self.integration_range, dims=(0,)
            ),  # integrate backwards in time, [1-eps, 0]
            atol=tolerance,
            rtol=tolerance,
            method="dopri5",
        )

        theta_0 = theta_and_div_0[:, :-1]
        divergence = theta_and_div_0[:, -1]
        log_prior = compute_log_prior(theta_0)

        return torch.Tensor((log_prior - divergence).detach())

    def sample_and_log_prob_batch(
        self,
        *context_data: torch.Tensor,
        batch_size: int | None = None,
        tolerance: float = 1e-7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns conditional samples and their likelihoods for a batch
        of contexts by solving the joint ODE forwards in time. This is
        more efficient than calling sample_batch and log_prob_batch
        separately.

        If d/dt [phi(t), f(t)] = rhs joint with initial conditions
        [theta_0, log p(theta_0)], where theta_0 ~ p_0(theta_0), then
        [phi(1), f(1)] = [theta_1, log p(theta_0) + log p_1(theta_1) -
        log p(theta_0)] = [theta_1, log p_1(theta_1)].
        """

        self.network.eval()

        if len(context_data) == 0 and batch_size is None:
            raise ValueError(
                "For unconditional sampling, the batch size needs to be set."
            )
        elif len(context_data) > 0:
            if batch_size is not None:
                raise ValueError(
                    "For conditional sampling, the batch_size can not be set"
                    " manually as it is automatically determined by the"
                    " context_data."
                )
            batch_size = len(context_data[0])

        assert isinstance(batch_size, int)

        theta_0 = self.sample_theta_0(batch_size)
        log_prior = compute_log_prior(theta_0)
        theta_and_div_init = torch.cat(
            (theta_0, log_prior.unsqueeze(1)), dim=1
        )

        _, theta_and_div_1 = odeint(
            lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, *context_data
            ),
            theta_and_div_init,
            self.integration_range,  # integrate forwards in time, [0, 1-eps]
            atol=tolerance,
            rtol=tolerance,
            method="dopri5",
        )

        theta_1, log_prob_1 = theta_and_div_1[:, :-1], theta_and_div_1[:, -1]

        return theta_1, log_prob_1

    @property
    def integration_range(self) -> torch.Tensor:
        """
        Integration range for ODE: We integrate from `0` to `1 - eps`.
        For score matching, `eps > 0` is required for stability. For
        flow matching, we can also have `eps = 0`.
        """
        # return torch.FloatTensor([0.0, 1.0 - self.eps], device=self.device)
        return torch.tensor([0.0, 1.0 - self.eps], device=self.device).float()


def compute_divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the divergence. TODO: Of what exactly?
    """
    div: float | torch.Tensor = 0.0
    with torch.enable_grad():
        y.requires_grad_(True)
        x.requires_grad_(True)
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(
                y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
            )[0][..., i : i + 1]
        return torch.Tensor(div)


def compute_log_prior(theta_0: torch.Tensor) -> torch.Tensor:
    """
    Compute the log prior of theta_0 under the base distribution
    """
    N = theta_0.shape[1]
    return torch.Tensor(
        -N / 2.0 * float(np.log(2 * np.pi))
        - torch.sum(theta_0**2, dim=1) / 2.0
    )
