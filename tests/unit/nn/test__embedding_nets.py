"""
Tests for embedding_nets.py
"""

import pytest
import torch

from fm4ar.nn.embedding_nets import (
    PositionalEncoding,
    create_embedding_net,
)


def test__create_embedding_net() -> None:
    """
    Test `create_embedding_net()`.
    """

    # Case 1: Check for supports_dict_input=True
    block_configs = [
        {"block_type": "PositionalEncoding", "kwargs": {"n_freqs": 3}}
    ]
    create_embedding_net(  # This should work
        input_shape=(123,),
        block_configs=block_configs,
        supports_dict_input=False,
    )
    with pytest.raises(ValueError) as value_error:
        create_embedding_net(  # This should fail
            input_shape=(123,),
            block_configs=block_configs,
            supports_dict_input=True,
        )
    assert "The first block must be a `SupportsDictInput`!" in str(value_error)

    # Case 2: Standard use case
    block_configs = [
        {"block_type": "SoftClipFlux", "kwargs": {"bound": 10.0}},
        {"block_type": "Concatenate", "kwargs": {"keys": ["flux", "wlen"]}},
        {
            "block_type": "DenseResidualNet",
            "kwargs": {
                "output_dim": 5,
                "hidden_dims": (10, 10, 10),
            },
        },
    ]
    embedding_net, output_dim = create_embedding_net(
        input_shape=(123,),
        block_configs=block_configs,
        supports_dict_input=True,
    )
    assert isinstance(embedding_net, torch.nn.Sequential)
    assert len(embedding_net) == 3
    assert output_dim == 5
    assert embedding_net[2].initial_layer.in_features == 246
    dummy_input = {"flux": torch.randn(17, 123), "wlen": torch.randn(17, 123)}
    assert embedding_net(dummy_input).shape == (17, 5)


@pytest.mark.parametrize(
    "theta_dim, n_freqs, encode_theta",
    [
        (5, 1, True),
        (13, 3, False),
    ],
)
def test__positional_encoding(
    theta_dim: int,
    n_freqs: int,
    encode_theta: bool,
) -> None:
    """
    Test `PositionalEncoding`.
    """

    # Create a positional encoding module
    positional_encoding = PositionalEncoding(
        n_freqs=n_freqs,
        encode_theta=encode_theta,
    )

    # Create a batch with random input
    batch_size = 17
    t_theta = torch.randn(batch_size, 1 + theta_dim)
    encoded = positional_encoding(t_theta)

    # Check that the output has the correct shape
    assert encoded.shape == (
        batch_size,
        1 + theta_dim + 2 * (1 + int(encode_theta) * theta_dim) * n_freqs,
    )
