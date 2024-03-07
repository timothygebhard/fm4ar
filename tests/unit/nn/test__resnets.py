"""
Tests for `fm4ar.nn.resnets`.
"""

import pytest
import torch

from fm4ar.nn.resnets import InitialLayerForZeroInputs, DenseResidualNet


def test__initial_layer_for_zero_inputs() -> None:
    """
    Test `fm4ar.nn.resnets.InitialLayerForZeroInputs`.
    """

    layer = InitialLayerForZeroInputs(output_dim=5)
    assert torch.equal(
        layer(torch.zeros(17, 0)),
        torch.zeros(17, 5),
    )


def test__dense_residual_net() -> None:
    """
    Test `fm4ar.nn.resnets.DenseResidualNet`.
    """

    # Case 1: Illegal input shape
    with pytest.raises(ValueError) as value_error:
        DenseResidualNet(input_shape=(1, 2, 3), output_dim=5, hidden_dims=())
    assert "DenseResidualNet only supports 1D inputs!" in str(value_error)

    # Case 2: input_dim != 0
    net = DenseResidualNet(
        input_shape=(3,),
        output_dim=5,
        hidden_dims=(7, 11, 13),
        activation="ReLU",
        dropout=0.13,
        use_batch_norm=True,
        use_layer_norm=False,
        context_features=17,
    )
    out = net(x=torch.randn(19, 3), context=torch.randn(19, 17))
    assert out.shape == (19, 5)

    # Case 3: input_dim == 0
    net = DenseResidualNet(
        input_shape=(0,),
        output_dim=5,
        hidden_dims=(7,),
        activation="ReLU",
        dropout=0.0,
        use_batch_norm=False,
        use_layer_norm=True,
        context_features=17,
    )
    out = net(x=torch.randn(19, 0), context=torch.randn(19, 17))
    assert out.shape == (19, 5)

    # Case 4: no context features
    net = DenseResidualNet(
        input_shape=(3,),
        output_dim=5,
        hidden_dims=(7,),
    )
    out = net(x=torch.randn(19, 3))
    assert out.shape == (19, 5)
