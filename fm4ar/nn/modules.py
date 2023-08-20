"""
This file contains the `Module` version of various functions, such as
the `mean()`, so that they can be used in a `nn.Sequential` container.
"""

import torch


class Rescale(torch.nn.Module):

    def __init__(
        self,
        lambda_min: float = 0.95,
        lambda_max: float = 2.45,
    ) -> None:
        super().__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.lambda_min) / (self.lambda_max - self.lambda_min)


class Mean(torch.nn.Module):

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Unsqueeze(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=self.dim)


class Tile(torch.nn.Module):

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.tile(self.shape)
