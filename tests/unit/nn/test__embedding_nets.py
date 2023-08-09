"""
Tests for embedding_nets.py
"""

from pathlib import Path

import pytest
import torch

from fm4ar.nn.embedding_nets import (
    PositionalEncoding,
    PrecomputedPCAEmbedding,
)


@pytest.fixture
def path_to_precomputed_pca_file(tmp_path: Path) -> Path:
    """
    Create a temporary file containing pre-computed dummy PCA weights.
    """

    data = {
        "mean": torch.randn(42).numpy(),
        "components": torch.randn(42, 42).numpy(),
    }

    file_path = tmp_path / "precomputed_pca.pt"
    torch.save(data, file_path)

    return file_path


def test__precomputed_pca_embedding(
    path_to_precomputed_pca_file: Path,
) -> None:
    """
    Test `PrecomputedPCAEmbeddingz.
    """

    batch_size = 17
    n_components = 10
    input_dim = 42

    embedding_net = PrecomputedPCAEmbedding(
        file_path=path_to_precomputed_pca_file.as_posix(),
        n_components=n_components,
        subtract_mean=True,
        freeze_weights=True,
    )

    # Check that the file contents are loaded correctly
    assert embedding_net.mean.shape == (input_dim,)
    assert embedding_net.components.shape == (input_dim, input_dim)

    # Check that the weights are frozen
    assert not embedding_net.mean.requires_grad
    assert not embedding_net.components.requires_grad

    # Check that the embedding has the correct shape
    x = torch.randn(batch_size, input_dim)
    embedding = embedding_net(x)
    assert embedding.shape == (batch_size, n_components)


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
