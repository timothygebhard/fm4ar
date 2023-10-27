"""
FNet implementation (less memory-hungry alternative to transformers).
"""

import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, expansion_factor * input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * input_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(self.layers(x))


def fourier_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(torch.fft.fft2(x, dim=(-1, -2)).real)


class FNetEncoderLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        expansion_factor:  int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.feed_forward = FeedForward(input_dim, expansion_factor, dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim, eps=1e-12)
        self.layer_norm_2 = nn.LayerNorm(input_dim, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x
        x = fourier_transform(x)
        x = self.layer_norm_1(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm_2(x + residual)

        return torch.Tensor(x)


class FNet(nn.Module):

    def __init__(
        self,
        input_dim: int = 256,
        expansion_factor: int = 2,
        dropout: float = 0.5,
        n_blocks: int = 6,
    ) -> None:

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                FNetEncoderLayer(input_dim, expansion_factor, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        return torch.Tensor(x)
