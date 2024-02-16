"""
Unit tests for `fm4ar.nn.modules`.
"""

import torch

from fm4ar.nn.modules import (
    Mean,
    Rescale,
    Sine,
    Tile,
    Unsqueeze,
)


def test__mean() -> None:
    """
    Test `Mean`.
    """

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    mean = Mean(dim=1)
    assert torch.allclose(mean(x), torch.mean(x, dim=1))

    mean = Mean(dim=0)
    assert torch.allclose(mean(x), torch.mean(x, dim=0))


def test__rescale() -> None:
    """
    Test `Rescale`.
    """

    x = torch.tensor([0.95, 1.70, 2.45])

    rescale = Rescale(lambda_min=0.95, lambda_max=2.45)
    assert torch.allclose(rescale(x), torch.tensor([0.0, 0.5, 1.0]))


def test__sine() -> None:
    """
    Test `Sine`.
    """

    x = torch.tensor([0.0, 1.0, 2.0])

    sine = Sine(w0=1.0)
    assert torch.allclose(sine(x), torch.sin(x))
    assert str(sine) == "Sine(w0=1.0)"


def test__tile() -> None:
    """
    Test `Tile`.
    """

    x = torch.tensor([1.0, 2.0, 3.0])

    tile = Tile(shape=(2, 1))
    assert torch.allclose(tile(x), torch.tile(x, (2, 1)))


def test__unsqueeze() -> None:
    """
    Test `Unsqueeze`.
    """

    x = torch.tensor([1.0, 2.0, 3.0])

    unsqueeze = Unsqueeze(dim=0)
    assert torch.allclose(unsqueeze(x), torch.unsqueeze(x, dim=0))

    unsqueeze = Unsqueeze(dim=1)
    assert torch.allclose(unsqueeze(x), torch.unsqueeze(x, dim=1))
