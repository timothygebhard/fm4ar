"""
Methods for scaling the target parameters (theta).
"""

from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER


class Scaler:
    """
    Base class for scaling the target parameters (theta).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the given tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse the scaling of the given tensor.
        """
        raise NotImplementedError


class IdentityScaler(Scaler):
    """
    Identity scaler (in case we want to disable scaling).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Standardizer(Scaler):
    """
    Standardize the target parameters (theta) by subtracting the mean
    and dividing by the standard deviation.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()

        self.mean: torch.Tensor = mean.float()
        self.std: torch.Tensor = std.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Normalizer(Scaler):
    """
    Normalize the target parameters (theta) by scaling them to the
    interval [0, 1].
    """

    def __init__(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
    ) -> None:
        super().__init__()

        self.minimum: torch.Tensor = minimum.float()
        self.maximum: torch.Tensor = maximum.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minimum) / (self.maximum - self.minimum)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.maximum - self.minimum) + self.minimum


def get_theta_scaler(config: dict[str, Any]) -> Scaler:
    """
    Get the scaler for theta specified in the given configuration.
    """

    # Get the dataset name and the scaler type
    name = config["data"]["name"]
    mode = config["data"]["theta_scaler"]

    # Set up the scaler
    scaler: Scaler
    if mode == "standardizer":
        mean, std = get_mean_and_std(name=name)
        scaler = Standardizer(mean=mean, std=std)
    elif mode == "normalizer":
        minimum, maximum = get_min_and_max(name=name)
        scaler = Normalizer(minimum=minimum, maximum=maximum)
    elif mode == "identity":
        scaler = IdentityScaler()
    else:
        raise ValueError(f"Unknown scaler mode: {mode}")

    return scaler


def get_mean_and_std(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the mean and standard deviation of the target parameters.
    """

    if name == "vasist-2023":
        a = torch.from_numpy(np.array(LOWER)).float()
        b = torch.from_numpy(np.array(UPPER)).float()
        mean = (a + b) / 2
        std = torch.sqrt(1 / 12 * (b - a) ** 2)

    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    return mean, std


def get_min_and_max(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the minimum and maximum of the target parameters.
    """

    if name == "vasist-2023":
        minimum = torch.from_numpy(np.array(LOWER)).float()
        maximum = torch.from_numpy(np.array(UPPER)).float()

    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    return minimum, maximum
