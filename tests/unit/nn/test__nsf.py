"""
Tests for `fm4ar.nn.nsf`.
"""

import pytest
import torch

from fm4ar.nn.nsf import (
    create_base_transform,
    create_linear_transform,
    create_transform,
)


def test__create_base_transform() -> None:
    """
    Test `fm4ar.nn.nsf.create_base_transform()`.
    """

    # Case 1: rq-coupling, param_dim=1
    base_transform = create_base_transform(
        i=3,
        param_dim=1,
        context_dim=7,
        base_transform_type="rq-coupling",
    )
    out = base_transform(inputs=torch.randn(11, 1), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 1)
    assert out[1].shape == (11,)

    # Case 2: rq-coupling, param_dim>1
    base_transform = create_base_transform(
        i=3,
        param_dim=5,
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
        param_dim=5,
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
        create_base_transform(i=3, param_dim=5, base_transform_type="invalid")


def test__create_linear_transform() -> None:
    """
    Test `fm4ar.nn.nsf.create_linear_transform()`.
    """

    linear_transform = create_linear_transform(param_dim=5)
    out = linear_transform(torch.randn(10, 5))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (10, 5)
    assert out[1].shape == (10,)


def test__create_transform() -> None:
    """
    Test `fm4ar.nn.nsf.create_transform()`.
    """

    transform = create_transform(
        num_flow_steps=3,
        theta_dim=5,
        context_dim=7,
        base_transform_kwargs=dict(
            base_transform_type="rq-coupling",
        ),
    )
    out = transform(inputs=torch.randn(11, 5), context=torch.randn(11, 7))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (11, 5)
    assert out[1].shape == (11,)
