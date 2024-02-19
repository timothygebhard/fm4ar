"""
Methods for creating normalizing flows.

This provides wrappers around two different implementations / libraries:

  - normflows
  - glasflow (which is in turn built on top of nflows)

They support slightly different sets of features (requiring adjustments
to the experiment config files) and give somewhat different performance.
[When training on an H100 GPU, glasflow seems about 1.5x faster.]

The glasflow-based NSF implementation is mostly based on the uci.py
example from https://github.com/bayesiains/nsf.
"""

from typing import Any, Type

import torch

from glasflow.nflows import distributions, flows, transforms, utils
from glasflow.nflows.nn import nets as nflows_nets
import normflows as nf

from fm4ar.utils.torchutils import (
    get_activation_from_name,
    load_and_or_freeze_model_weights,
)


# -----------------------------------------------------------------------------
# Unified interface for both
# -----------------------------------------------------------------------------


class FlowWrapper(torch.nn.Module):
    """
    This is a thin wrapper around the different flow implementations
    that handles different conventions, e.g., for `.sample()`.

    Note: This inherits from `torch.nn.Module` so calling `.to()` on
    the model will also move the `flow` to the correct device.
    """

    def __init__(self, flow: nf.NormalizingFlow | flows.Flow) -> None:
        super().__init__()
        self.flow = flow

    def sample(
        self,
        num_samples: int,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample from the flow model.

        Args:
            num_samples: Number of samples to draw.
            context: Context tensor (if applicable). This needs to be
                 a tensor, NOT a dict.

        Returns:
            Samples from the flow model.
        """

        if isinstance(self.flow, nf.ConditionalNormalizingFlow):
            samples, _ = self.flow.sample(
                num_samples=num_samples if context is None else len(context),
                context=context,
            )
        elif isinstance(self.flow, nf.NormalizingFlow):
            samples, _ = self.flow.sample(num_samples=num_samples)
        elif isinstance(self.flow, flows.Flow):
            samples = self.flow.sample(
                num_samples=num_samples,
                context=context,
            )
            samples = samples.squeeze(1)
        else:  # pragma: no cover
            raise ValueError(f"Unknown flow type: {type(self.flow)}")

        return torch.Tensor(samples)

    def log_prob(
        self,
        theta: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the log probability of the given samples.

        Args:
            theta: Samples for which to compute the log probability.
            context: Context tensor (if applicable). This needs to be
                a tensor, NOT a dict.

        Returns:
            Log probabilities of the `theta`.
        """

        if isinstance(self.flow, nf.ConditionalNormalizingFlow):
            log_prob = self.flow.log_prob(x=theta, context=context)
        elif isinstance(self.flow, nf.NormalizingFlow):
            log_prob = self.flow.log_prob(x=theta)
        elif isinstance(self.flow, flows.Flow):
            log_prob = self.flow.log_prob(inputs=theta, context=context)
        else:  # pragma: no cover
            raise ValueError(f"Unknown flow type: {type(self.flow)}")

        return torch.Tensor(log_prob)

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model and compute the log probability.

        Args:
            num_samples: Number of samples to draw.
            context: Context tensor (if applicable). This needs to be
                a tensor, NOT a dict.

        Returns:
            A 2-tuple with the samples and the log probabilities.
        """

        if isinstance(self.flow, nf.ConditionalNormalizingFlow):
            samples, log_prob = self.flow.sample(
                num_samples=num_samples if context is None else len(context),
                context=context,
            )
        elif isinstance(self.flow, nf.NormalizingFlow):
            samples, log_prob = self.flow.sample(num_samples=num_samples)
        elif isinstance(self.flow, flows.Flow):
            samples, log_prob = self.flow.sample_and_log_prob(
                num_samples=num_samples,
                context=context,
            )
            samples = samples.squeeze(1)
            log_prob = log_prob.squeeze(1)
        else:  # pragma: no cover
            raise ValueError(f"Unknown flow type: {type(self.flow)}")

        return torch.Tensor(samples), torch.Tensor(log_prob)


def create_flow_wrapper(
    dim_theta: int,
    dim_context: int,
    flow_wrapper_config: dict[str, Any],
) -> FlowWrapper:
    """
    Create a normalizing flow model based on the given configuration,
    which is wrapped in a thin compatibility layer (FlowWrapper).

    This is the analogon of `create_vectorfield_net()` for FMPE.

    Args:
        dim_theta: Dimensionality of `theta`.
        dim_context: Dimensionality of the `context`.
        flow_wrapper_config: Configuration for the discrete
            flow wrapper (i.e., the "discrete_flow_wrapper_config"
            section inside the "model" part of the experiment config).

    Returns:
        FlowWrapper: A thin wrapper around the flow model.
    """

    # Define some shortcuts
    flow_library = flow_wrapper_config["flow_library"]
    flow_kwargs = flow_wrapper_config["kwargs"]
    freeze_weights = flow_wrapper_config.get("freeze_weights", False)
    load_weights = flow_wrapper_config.get("load_weights", {})

    # Construct the flow model based on the specified library
    match flow_library:
        case "glasflow":
            flow_wrapper = create_glasflow_flow(
                dim_theta=dim_theta,
                dim_context=dim_context,
                flow_kwargs=flow_kwargs,
            )
        case "normflows":
            flow_wrapper = create_normflows_flow(
                dim_theta=dim_theta,
                dim_context=dim_context,
                flow_kwargs=flow_kwargs,
            )
        case _:  # pragma: no cover
            raise ValueError(f"Unknown flow library: {flow_library}")

    # Load pre-trained weights or freeze the weights of the flow
    load_and_or_freeze_model_weights(
        model=flow_wrapper.flow,
        freeze_weights=freeze_weights,
        load_weights=load_weights,
    )

    return flow_wrapper


# -----------------------------------------------------------------------------
# normflows
# -----------------------------------------------------------------------------


def create_normflows_flow(
    dim_theta: int,
    dim_context: int | None,
    flow_kwargs: dict[str, Any],
) -> FlowWrapper:
    """
    Create a normflows-based normalizing flow.
    """

    # Define shortcuts
    num_flow_steps = flow_kwargs["num_flow_steps"]
    base_transform_type = flow_kwargs["base_transform_type"]
    base_transform_kwargs = flow_kwargs["base_transform_kwargs"]

    # We need to copy the base_transform_kwargs because we will modify the
    # activation function and we don't want to change the original config
    # as this will cause issues when resuming from a checkpoint.
    base_transform_kwargs = base_transform_kwargs.copy()

    # Update the activation function: the config uses a string, but normflows
    # expects a class like torch.nn.ReLU (*not* an instance!)
    base_transform_kwargs["activation"] = get_activation_from_name(
        base_transform_kwargs["activation"]
        ).__class__

    # Set base transform
    if base_transform_type == "rq-coupling":
        BaseTransform = nf.flows.CoupledRationalQuadraticSpline
    elif base_transform_type == "rq-autoregressive":
        BaseTransform = nf.flows.AutoregressiveRationalQuadraticSpline
    else:
        raise ValueError(f"Unknown base transform type: {base_transform_type}")

    # Construct flow steps
    flows = []
    for _ in range(num_flow_steps):
        flows += [
            BaseTransform(
                num_input_channels=dim_theta,
                num_context_channels=dim_context,
                **base_transform_kwargs,
            )
        ]
        flows += [nf.flows.LULinearPermute(dim_theta)]

    # Set base distribution
    q0 = nf.distributions.DiagGaussian(dim_theta, trainable=False)

    # Construct flow model
    flow = nf.ConditionalNormalizingFlow(q0=q0, flows=flows)

    return FlowWrapper(flow=flow)


# -----------------------------------------------------------------------------
# glasflow
# -----------------------------------------------------------------------------


def create_glasflow_flow(
    dim_theta: int,
    dim_context: int | None,
    flow_kwargs: dict[str, Any],
) -> FlowWrapper:
    """
    Create a glasflow-based normalizing flow.
    """

    # Define series of transforms
    transform = create_transform(
        theta_dim=dim_theta,
        context_dim=dim_context,
        **flow_kwargs,
    )

    # Define base distribution
    distribution = distributions.StandardNormal((dim_theta,))

    # We set the embedding net to the identity, because we handle the context
    # embedding separately in the `NPENetwork` wrapper.
    flow = flows.Flow(
        transform=transform,
        distribution=distribution,
        embedding_net=torch.nn.Identity(),
    )

    return FlowWrapper(flow=flow)


def create_linear_transform(param_dim: int) -> transforms.CompositeTransform:
    """
    Create the composite linear transform PLU.

    Args:
        param_dim: Dimension of the parameter space.

    Returns:
        The linear transform PLU.
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_base_transform(
    i: int,
    theta_dim: int,
    context_dim: int | None = None,
    hidden_dim: int = 512,
    num_transform_blocks: int = 2,
    activation: str = "ReLU",
    dropout_probability: float = 0.0,
    batch_norm: bool = False,
    num_bins: int = 8,
    tail_bound: float = 1.0,
    apply_unconditional_transform: bool = False,
    base_transform_type: str = "rq-coupling",
) -> transforms.Transform:
    """
    Build a base NSF transform of theta (parameters), conditioned on x.

    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as
    described in the Neural Spline Flow paper (arXiv:1906.04032).

    A coupling flow fixes half the components of y, and applies a
    transform to the remaining components, conditioned on the fixed
    components. This is a restricted form of an autoregressive
    transform, with a single split into fixed / transformed components.

    The transform here is a neural spline flow, where the flow is
    parametrized by a residual neural network that depends on
    theta_fixed and x. The residual network consists of a sequence of
    two-layer fully-connected blocks.

    Args:
        i: Index of transform in sequence; needed for alternating masks.
        theta_dim: Number of parameters.
        context_dim: Dimensionality of context.
        hidden_dim: Number of hidden units per layer.
        num_transform_blocks: Number of transform blocks comprising
            the transform.
        activation: Activation function as a string (e.g., "relu").
        dropout_probability: Dropout probability for regularization.
        batch_norm: Whether to use batch normalization.
        num_bins: Number of bins for the spline.
        tail_bound: Tail bound. This is the maximum absolute value of
            the input to the inverse CDF of the spline; values outside
            this range are not modified by the transform.
        apply_unconditional_transform: Whether to apply an unconditional
            transform to fixed components.
        base_transform_type: Type of base transform, must be one of
            {rq-coupling, rq-autoregressive}.

    Returns:
        The NSF transform.
    """

    activation_fn = get_activation_from_name(activation)

    if base_transform_type == "rq-coupling":
        if theta_dim == 1:
            mask = torch.tensor([1], dtype=torch.uint8)
        else:
            mask = utils.create_alternating_binary_mask(
                theta_dim, even=(i % 2 == 0)
            )
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=(
                lambda in_features, out_features: nflows_nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_transform_blocks,
                    activation=activation_fn,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm,
                )
            ),
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )

    elif base_transform_type == "rq-autoregressive":
        return (
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=theta_dim,
                hidden_features=hidden_dim,
                context_features=context_dim,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                num_blocks=num_transform_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=activation_fn,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm,
            )
        )

    else:
        raise ValueError(f"Unknown base transform type: {base_transform_type}")


def create_transform(
    theta_dim: int,
    context_dim: int | None,
    num_flow_steps: int,
    base_transform_type: str,
    base_transform_kwargs: dict,
) -> transforms.Transform:
    """
    Build a sequence of NSF transforms, which maps parameters `theta`
    into the base distribution (usually a Gaussian). Transforms are
    conditioned on a given context.

    Note that the forward map is `f^{-1}(theta, x)`, where `theta` are
    the parameters and `x` is the context.

    Each step in the sequence consists of
        * A linear transform of theta, which in particular permutes
          components
        * A NSF transform of theta, conditioned on x.
    There is one final linear transform at the end.

    Args:
        theta_dim: Number of parameters (dimensionality of the flow).
        context_dim: Dimensionality of the (embedded) context.
        num_flow_steps: Number of transforms in sequence.
        base_transform_type: Type of base transform, must be one of
            {rq-coupling, rq-autoregressive}.
        base_transform_kwargs: Keyword arguments for the base transform.

    Returns:
        The NSF transform sequence.
    """

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(theta_dim),
                    create_base_transform(
                        i=i,
                        theta_dim=theta_dim,
                        context_dim=context_dim,
                        base_transform_type=base_transform_type,
                        **base_transform_kwargs,
                    ),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(theta_dim)]
    )

    return transform


def create_unconditional_nsf(
    num_transforms: int = 24,
    num_input_channels: int = 16,
    num_hidden_channels: int = 512,
    num_blocks: int = 4,
    num_bins: int = 16,
    tail_bound: float = 5.0,
    activation: Type[torch.nn.Module] = torch.nn.ELU,
) -> nf.NormalizingFlow:
    """
    Create an unconditional neural spline flow model (with normflows).

    This is useful, e.g., to fit samples from a posterior so that one
    can evaluate the logprob and use it for importance sampling.
    """

    # Construct series of transforms
    flows = []
    for _ in range(num_transforms):
        flows += [
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels=num_input_channels,
                num_blocks=num_blocks,
                num_hidden_channels=num_hidden_channels,
                num_bins=num_bins,
                tail_bound=tail_bound,
                activation=activation,
            ),
            nf.flows.LULinearPermute(num_input_channels),
        ]

    # Set base distribution
    q0 = nf.distributions.DiagGaussian(num_input_channels, trainable=False)

    # Construct flow model
    model = nf.NormalizingFlow(q0=q0, flows=flows)

    return model
