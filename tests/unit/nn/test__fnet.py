"""
Unit tests for `fm4ar.nn.fnet`.
"""

import torch

from fm4ar.nn.fnet import FNet


def test__fnet() -> None:
    """
    Test `fm4ar.nn.fnet.FNet`.
    """

    batch_size = 10
    n_bins = 17
    latent_dim = 5

    fnet = FNet(
        input_dim=latent_dim,
        n_blocks=2,
        expansion_factor=2,
        dropout=0.1,
    )

    x = torch.randn(batch_size, n_bins, latent_dim)
    y = fnet(x)

    assert y.shape == (batch_size, n_bins, latent_dim)
