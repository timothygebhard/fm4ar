"""
Define `FlowMatching` model.
"""

from typing import Any

import numpy as np
import torch
from torchdiffeq import odeint

from fm4ar.models.continuous.model import create_cf_model
from fm4ar.models.base import Base


class FlowMatching(Base):
    """
    Class for continuous normalizing flows trained with flow matching.

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

        t         ~ U[0, 1)                             (noise level)
        theta_0   ~ N(0, 1)                             (sampled noise)
        theta_1   = theta                               (pure sample)
        theta_t   = c1(t) * theta_1 + c0(t) * theta_0   (noisy sample)

    where the coefficients `c0(t)` and `c1(t)` are defined as:

        c0        = (1 - (1 - sigma_min) * t)
        c1        = t

    and the final training objective is given by:

        v_target  = theta_1 - (1 - sigma_min) * theta_0
        loss      = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)

        model_config = self.config["model"]

        self.sigma_min = model_config["sigma_min"]
        self.time_prior_exponent = model_config.get("time_prior_exponent", 0.)
        self.theta_dim = model_config["theta_dim"]

    def evaluate_vectorfield(
        self,
        t: float | torch.Tensor,
        theta_t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Evaluate the vectorfield at the given time and parameter values.

        Args:
            t: The time at which to evaluate the vectorfield (must be
                a float between 0 and 1). Note: This can also be a
                tensor in case we are evaluating the vectorfield for
                a batch.
            theta_t: The parameter values at which to evaluate the
                vectorfield.
            context: Context (i.e., observed data).

        Returns:
            The vectorfield evaluated at the given time and parameter.
        """

        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)

        return torch.Tensor(self.model(t=t, theta=theta_t, context=context))

    def initialize_model(self) -> None:
        """
        Initialize the neural net that parameterizes the vectorfield.
        """

        self.model = create_cf_model(model_kwargs=self.config["model"])

    @property
    def integration_range(self) -> torch.Tensor:
        """
        Integration range for ODE solver: [0, 1].
        """

        return torch.tensor([0.0, 1.0], device=self.device).float()

    def log_prob_batch(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
        tolerance: float = 1e-7,
        method: str = "dopri5",
    ) -> torch.Tensor:
        """
        Evaluates log_probs of theta conditional on provided context.
        For this we solve an ODE backwards in time until we reach the
        initial pure noise distribution.

        There are two contributions, the log_prob of theta_0 (which is
        uniquely determined by theta) under the base distribution, and
        the integrated divergence of the vectorfield.

        Args:
            theta: Parameter values for which to evaluate the log_prob.
            context: Context (i.e., observed data).
            tolerance: Tolerance (atol and rtol) for the ODE solver.
            method: ODE solver method. Default is "dopri5".

        Returns:
            The log probability of `theta`.
        """

        self.model.eval()

        div_init = torch.zeros(
            (theta.shape[0],), device=theta.device
        ).unsqueeze(1)
        theta_and_div_init = torch.cat((theta, div_init), dim=1)

        # Integrate backwards in time to get from theta_1 to theta_0;
        # note the `flip()` of the integration range
        _, theta_and_div_0 = odeint(
            func=lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, context
            ),
            y0=theta_and_div_init,
            t=torch.flip(self.integration_range, dims=(0,)),
            atol=tolerance,
            rtol=tolerance,
            method=method,
        )

        theta_0 = theta_and_div_0[:, :-1]
        divergence = theta_and_div_0[:, -1]
        log_prior = compute_log_prior(theta_0)

        return torch.Tensor((log_prior - divergence).detach())

    def loss(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculates loss as the the mean squared error between the
        predicted vectorfield and the vector field for transporting the
        parameter data to samples from the prior.

        Args:
            theta: Parameters.
            context: Context (i.e., observed data).

        Returns:
            Loss tensor.
        """

        # Get the time prior exponent from the kwargs, if provided. This can
        # be used, e.g., to increase the time prior exponent during training.
        time_prior_exponent = kwargs.get("time_prior_exponent", None)

        # Sample a time t and some starting parameters theta_0
        t = self.sample_t(
            num_samples=len(theta),
            time_prior_exponent=time_prior_exponent
        )
        theta_0 = self.sample_theta_0(num_samples=len(theta))

        # Use optimal transport path to interpolate theta_t between theta_0
        # (starting values) and theta_1 (target parameters)
        theta_t = ot_conditional_flow(
            theta_0=theta_0,
            theta_1=theta,
            t=t,
            sigma_min=self.sigma_min,
        )

        # Compute the true vectorfield and the predicted vectorfield
        true_vf = theta - (1 - self.sigma_min) * theta_0
        pred_vf = self.model(t=t, theta=theta_t, context=context)

        # Calculate loss as MSE between the true and predicted vectorfield
        loss = torch.nn.functional.mse_loss(pred_vf, true_vf)

        return torch.Tensor(loss)

    def rhs_of_joint_ode(
        self,
        t: float,
        theta_and_div_t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Returns the right hand side of the neural ODE that is used to
        evaluate the log_prob of theta samples. This is a joint ODE over
        the vectorfield and the divergence. By integrating this ODE, one
        can simultaneously trace the parameter sample theta_t and
        integrate the divergence contribution to the log_prob, see e.g.,
        https://arxiv.org/abs/1806.07366 or Appendix C in
        https://arxiv.org/abs/2210.02747.

        Args:
            t: Time (controls the noise level).
            theta_and_div_t: Concatenated tensor of `(theta_t, div)`.
            context: Context (i.e., observed data).

        Returns:
            The vector field that generates the continuous flow, plus
            its divergence (required for likelihood evaluation).
        """

        theta_t = theta_and_div_t[:, :-1]  # extract theta_t
        with torch.enable_grad():
            theta_t.requires_grad_(True)
            vf = self.evaluate_vectorfield(t, theta_t, context)
            div_vf = compute_divergence(vf, theta_t)
        return torch.cat((vf, -div_vf), dim=1)

    def sample_and_log_prob_batch(
        self,
        context: torch.Tensor | None,
        num_samples: int = 1,
        tolerance: float = 1e-7,
        method: str = "dopri5",
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

        Args:
            context: Context (i.e., observed data).
            num_samples: Number of posterior samples to generate in the
                unconditional case. If `context` is provided, the number
                of samples is automatically determined from the context.
            tolerance: Tolerance (atol and rtol) for the ODE solver.
            method: ODE solver method. Default is "dopri5".

        Returns:
            The generated samples and their log probabilities.
        """

        self.model.eval()

        # Get the number of samples, either from context or explicitly
        num_samples = len(context) if context is not None else num_samples

        theta_0 = self.sample_theta_0(num_samples)
        log_prior = compute_log_prior(theta_0)
        theta_and_div_init = torch.cat(
            (theta_0, log_prior.unsqueeze(1)), dim=1
        )

        # Integrate forwards in time to get from theta_0 to theta_1
        _, theta_and_div_1 = odeint(
            func=lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, context
            ),
            y0=theta_and_div_init,
            t=self.integration_range,
            atol=tolerance,
            rtol=tolerance,
            method=method,
        )

        theta_1, log_prob_1 = theta_and_div_1[:, :-1], theta_and_div_1[:, -1]

        return theta_1, log_prob_1

    def sample_batch(
        self,
        context: torch.Tensor | None,
        num_samples: int = 1,
        tolerance: float = 1e-7,
        method: str = "dopri5",
    ) -> torch.Tensor:
        """
        Returns (conditional) samples for a batch of contexts by solving
        an ODE forwards in time.

        Args:
            context: Context (i.e., observed data).
            num_samples: Number of posterior samples to generate in the
                unconditional case. If `context` is provided, the number
                of samples is automatically determined from the context.
            tolerance: Tolerance (atol and rtol) for the ODE solver.
            method: ODE solver method. Default is "dopri5".

        Returns:
            The generated samples.
        """

        self.model.eval()

        # Get the number of samples, either from context or explicitly
        num_samples = len(context) if context is not None else num_samples

        # Solve ODE forwards in time to get from theta_0 to theta_1
        with torch.no_grad():
            theta_0 = self.sample_theta_0(num_samples)
            _, theta_1 = odeint(
                func=lambda t, theta_t: self.evaluate_vectorfield(
                    t, theta_t, context
                ),
                y0=theta_0,
                t=self.integration_range,
                atol=tolerance,
                rtol=tolerance,
                method=method,
            )

        return torch.Tensor(theta_1)

    def sample_t(
        self,
        num_samples: int,
        time_prior_exponent: float | None = None,
    ) -> torch.Tensor:
        """
        Sample time `t` (in [0, 1]) from a power law distribution.
        """

        # If time_prior_exponent is not provided, use the default value
        if time_prior_exponent is None:
            time_prior_exponent = self.time_prior_exponent

        # Sample t from a power law distribution
        # exponent = 0 corresponds to a uniform distribution (equal weights)
        # exponent = 1 corresponds to a linear distribution (more weight on 1)
        t = torch.rand(num_samples, device=self.device)
        t = torch.pow(t, 1 / (1 + time_prior_exponent))

        return t

    def sample_theta_0(self, num_samples: int) -> torch.Tensor:
        """
        Sample `theta_0` from a standard Gaussian prior.
        """

        return torch.randn(num_samples, self.theta_dim, device=self.device)


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
