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

import numpy as np

from fm4ar.datasets.noise import (
    NoiseGenerator,
    get_noise_transform_from_string,
)


class DataTransform:
    """
    Generic base class for all data transforms.

    Note: Unlike the `ThetaScaler` class, the data transforms typically
    do not have an inverse transformation, because adding noise or
    subsampling is not reversible.
    """

    @abstractmethod
    def forward(self, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply the forward transformation to the given tensor.
        """
        raise NotImplementedError  # pragma: no cover


def get_data_transforms(stage_config: dict) -> list[DataTransform]:
    """
    Build the data transforms for the given stage.

    Args:
        stage_config: Dictionary containing the config for a stage.

    Returns:
        List of data transforms.
    """

    data_transforms: list[DataTransform] = []

    # Loop over the data transforms and instantiate the corresponding classes
    for transform_config in stage_config.get("data_transforms", []):

        # Get the class and the keyword arguments
        method = transform_config["method"]
        kwargs = transform_config.get("kwargs", {})

        # Instantiate the class and append it to the list
        match method:
            case "add_noise":
                data_transforms.append(AddNoise(**kwargs))
            case "subsample":
                data_transforms.append(Subsample(**kwargs))
            case _:
                raise ValueError(f"Unknown data transform: {method}")

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

    def forward(self, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Add noise to the given flux.
        """

        # TODO: Maybe we want to thing about the "SNR" here, too?
        #   Example: Rescale the noise to a certain target SNR?

        # Sample the error bars; store them in the input dictionary
        x["error_bars"] = self.noise_generator.sample_error_bars(
            wlen=x["wlen"],
        )

        # Draw a noise realization and add it to the flux
        x["flux"] += self.noise_generator.sample_noise(
            error_bars=x["error_bars"],
        )

        return x


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

    def forward(self, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Subsample the given flux.
        """

        # Get the indices of the subsampled flux
        idx = self.rng.choice(
            np.arange(x["flux"].size),
            size=int(x["flux"].size * self.factor),
            replace=False,
        )

        # Subsample the flux and wavelength
        x["flux"] = x["flux"][idx]
        x["wlen"] = x["wlen"][idx]

        # Subsample the error bars (if available)
        if "error_bars" in x:
            x["error_bars"] = x["error_bars"][idx]

        return x
