"""
Define data transformations that will be applied inside the data loading
pipeline, such as "add noise" or "re-bin to a different resolution".

Note: The data transformations in this file are allowed to change
between the different stages of training (e.g., "train for X epochs
with no noise, then train for Y epochs with noise"). This does *NOT*
include the `theta` scaling transformations, which are fixed for the
entire training process, and are therefore defined in a separate module.
"""

from abc import abstractmethod
from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, Field

from fm4ar.datasets.noise import (
    NoiseGenerator,
    get_noise_transform_from_string,
)


class DataTransformConfig(BaseModel):
    """
    Configuration for a data transform.
    """

    type: str = Field(
        ...,
        description="Type of the data transform.",
    )
    kwargs: dict = Field(
        {},
        description="Keyword arguments for the data transform.",
    )


class DataTransform:
    """
    Generic base class for all data transforms.

    Note: Unlike the `ThetaScaler` class, the data transforms typically
    do not have an inverse transformation, because adding noise or
    subsampling is not reversible.
    """

    @abstractmethod
    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply the forward transformation to the given tensor.
        """
        raise NotImplementedError  # pragma: no cover


def get_data_transforms(
    data_transform_configs: list[DataTransformConfig],
) -> list[DataTransform]:
    """
    Build the data transforms for the given stage.

    Args:
        data_transform_configs: List of data transform configurations.

    Returns:
        List of data transforms.
    """

    data_transform: DataTransform
    data_transforms: list[DataTransform] = []

    # Loop over the data transforms and instantiate the corresponding classes
    for data_transform_config in data_transform_configs:
        match data_transform_config.type:
            case "AddNoise":
                data_transform = AddNoise(**data_transform_config.kwargs)
            case "Subsample":
                data_transform = Subsample(**data_transform_config.kwargs)
            case _:
                raise ValueError(
                    f"Unknown data transform: {data_transform_config.type}"
                )
        data_transforms.append(data_transform)

    return data_transforms


class AddNoise(DataTransform):
    """
    Use a `NoiseGenerator` to add noise to the given flux.
    """

    def __init__(
        self,
        random_seed: int,
        complexity: int,
        transform: str,
    ) -> None:
        """
        Initialize a new `AddNoise` transform.

        See the `NoiseGenerator` class for more details, especially the
        documentation of the `transform` argument.

        Args:
            random_seed: Seed for the random number generator.
            complexity: Complexity of the noise model.
            transform: String specifying the noise transform.
        """

        super().__init__()

        self.noise_generator = NoiseGenerator(
            random_seed=random_seed,
            complexity=complexity,
            transform=get_noise_transform_from_string(transform),
        )

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Add noise to the given flux.
        """

        # TODO: Maybe we want to thing about the "SNR" here, too?
        #   Example: Rescale the noise to a certain target SNR?

        output = dict(x)

        # Sample the error bars; store them in the output dictionary
        error_bars = self.noise_generator.sample_error_bars(wlen=x["wlen"])
        output["error_bars"] = error_bars

        # Draw a noise realization and add it to the flux
        noise = self.noise_generator.sample_noise(error_bars=error_bars)
        output["flux"] += noise

        return output


class Subsample(DataTransform):
    """
    Randomly subsample the given flux, keeping only a fraction of the
    original data points. Note: This is not the same as re-binning!
    """

    def __init__(
        self,
        factor: float,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize a new `Subsample` transform.

        Args:
            factor: Factor (in [0, 1]) by which to subsample the flux.
            random_seed: Seed for the random number generator.
        """

        super().__init__()

        self.factor = factor
        self.rng = np.random.default_rng(random_seed)

    def forward(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Subsample the given flux.
        """

        # Get the indices of the subsampled flux
        idx = self.rng.choice(
            np.arange(x["flux"].size),
            size=int(x["flux"].size * self.factor),
            replace=False,
        )

        # Create a new dictionary with the subsampled data
        output = dict(x)

        # Subsample the flux and wavelength
        output["flux"] = x["flux"][idx]
        output["wlen"] = x["wlen"][idx]

        # Subsample the error bars (if available)
        if "error_bars" in output:
            output["error_bars"] = x["error_bars"][idx]

        return output
