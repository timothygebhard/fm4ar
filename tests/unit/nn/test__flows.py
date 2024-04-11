"""
Tests for `fm4ar.nn.flows`.
"""

import normflows as nf
import pytest
import torch

from fm4ar.nn.flows import (
    create_base_transform,
    create_linear_transform,
    create_transform,
    create_unconditional_flow_wrapper,
)


def test__create_base_transform() -> None:
    """
    Test `create_base_transform()`.
    """

    # Case 1: rq-coupling, param_dim = 1
    base_transform = create_base_transform(
        i=3,
        theta_dim=1,
        context_dim=7,
        base_transform_type="rq-coupling",
    )
    out = base_transform(inputs=torch.randn(11, 1), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 1)
    assert out[1].shape == (11,)

    # Case 2: rq-coupling, param_dim > 1
    base_transform = create_base_transform(
        i=3,
        theta_dim=5,
        context_dim=7,
        base_transform_type="rq-coupling",
    )
    out = base_transform(inputs=torch.randn(11, 5), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 5)
    assert out[1].shape == (11,)

    # Case 3: rq-autoregressive
    base_transform = create_base_transform(
        i=3,
        theta_dim=5,
        context_dim=7,
        base_transform_type="rq-autoregressive",
    )
    out = base_transform(inputs=torch.randn(11, 5), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 5)
    assert out[1].shape == (11,)

    # Case 4: ValueError
    with pytest.raises(ValueError):
        create_base_transform(i=3, theta_dim=5, base_transform_type="invalid")


def test__create_linear_transform() -> None:
    """
    Test `create_linear_transform()`.
    """

    linear_transform = create_linear_transform(param_dim=5)
    out = linear_transform(torch.randn(10, 5))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (10, 5)
    assert out[1].shape == (10,)


def test__create_transform() -> None:
    """
    Test `create_transform()`.
    """

    transform = create_transform(
        num_flow_steps=3,
        theta_dim=5,
        context_dim=7,
        base_transform_type="rq-coupling",
        base_transform_kwargs={},
    )
    out = transform(inputs=torch.randn(11, 5), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 5)
    assert out[1].shape == (11,)


@pytest.mark.parametrize(
    "flow_library, kwargs",
    [
        (
            "glasflow",
            {
                "num_flow_steps": 3,
                "base_transform_type": "rq-autoregressive",
                "base_transform_kwargs": {
                    "hidden_dim": 64,
                    "num_transform_blocks": 2,
                    "activation": "ELU",
                    "dropout_probability": 0.1,
                    "use_batch_norm": True,
                    "num_bins": 10,
                },
            },
        ),
        (
            "normflows",
            {
                "num_flow_steps": 3,
                "base_transform_type": "rq-autoregressive",
                "base_transform_kwargs": {
                    "num_blocks": 2,
                    "num_hidden_channels": 64,
                    "num_bins": 10,
                    "tail_bound": 10,
                    "activation": "ELU",
                    "dropout_probability": 0.1,
                },
            },
        ),
    ],
)
def test__create_unconditional_flow_wrapper(
    flow_library: str,
    kwargs: dict,
) -> None:
    """
    Test `create_unconditional_flow_wrapper()`.
    """

    flow_wrapper = create_unconditional_flow_wrapper(
        dim_theta=5,
        flow_wrapper_config={
            "flow_library": flow_library,
            "kwargs": kwargs,
        },
    )

    if flow_library == "normflows":
        assert isinstance(flow_wrapper.flow, nf.NormalizingFlow)

    assert flow_wrapper.sample(num_samples=7).shape == (7, 5)
    assert flow_wrapper.log_prob(theta=torch.randn(7, 5)).shape == (7,)

    theta, logprob = flow_wrapper.sample_and_log_prob(num_samples=7)
    assert theta.shape == (7, 5)
    assert logprob.shape == (7,)
