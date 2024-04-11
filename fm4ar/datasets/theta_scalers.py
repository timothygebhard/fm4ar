"""
Methods for scaling the target parameters `theta`.

Note: The `theta` scaling is different from the `data_transforms`, as
the it remains fixed for the entire training process while the data
transforms can change between different stages of training.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER


class ThetaScaler(ABC):
    """
    Base class for all theta scalers.
    """

    @abstractmethod
    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return self.forward({"theta": x})["theta"]

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.forward_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )

    @abstractmethod
    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

    def inverse_array(self, x: np.ndarray) -> np.ndarray:
        return self.inverse({"theta": x})["theta"]

    def inverse_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.from_numpy(self.inverse_array(x.cpu().numpy()))
            .type_as(x)
            .to(x.device)
        )


class IdentityScaler(ThetaScaler):
    """
    Identity scaler for `theta`.
    """

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(x)


class MeanStdScaler(ThetaScaler):
    """
    Scale `theta` by subtracting the mean and dividing by the std. dev.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:

        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["theta"] = (x["theta"] - self.mean) / self.std
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["theta"] = x["theta"] * self.std + self.mean
        return output


class MinMaxScaler(ThetaScaler):
    """
    Scale `theta` by mapping it into the interval [0, 1].
    """

    def __init__(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> None:

        super().__init__()

        self.minimum = minimum
        self.maximum = maximum
        self.difference = self.maximum - self.minimum

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["theta"] = (x["theta"] - self.minimum) / self.difference
        return output

    def inverse(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = dict(x)
        output["theta"] = x["theta"] * self.difference + self.minimum
        return output


def get_theta_scaler(config: dict[str, Any]) -> ThetaScaler:
    """
    Get the scaler for theta specified in the given `config` (which is
    only the config for the feature scaler for theta, not the entire
    experiment config).
    """

    # Case 1: No feature scaling defined for theta
    if not config:
        return IdentityScaler()

    # Case 2: Feature scaling defined
    scaler: ThetaScaler
    method = config["method"]
    kwargs = config.get("kwargs", {})
    match method:
        case "mean_std" | "MeanStdScaler":
            mean, std = get_mean_and_std(**kwargs)
            scaler = MeanStdScaler(mean=mean, std=std)
        case "min_max" | "MinMaxScaler":
            minimum, maximum = get_min_and_max(**kwargs)
            scaler = MinMaxScaler(minimum=minimum, maximum=maximum)
        case "identity" | "IdentityScaler":
            scaler = IdentityScaler()
        case _:
            raise ValueError(f"Unknown feature scaling method: {method}")

    return scaler


def get_mean_and_std(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and standard deviation of the target parameters.
    """

    if dataset == "vasist_2023":
        a = np.array(LOWER)
        b = np.array(UPPER)
        mean = (a + b) / 2
        std = np.sqrt(1 / 12 * (b - a) ** 2)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return mean, std


def get_min_and_max(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the minimum and maximum of the target parameters.
    """

    if dataset == "vasist_2023":
        minimum = np.array(LOWER)
        maximum = np.array(UPPER)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return minimum, maximum
