"""
Methods for generating noise that can be added to the target spectrum.
"""

from abc import ABC, abstractmethod

import numpy as np


class NoiseGenerator(ABC):
    """
    Abstract base class for noise generators (mostly for type hinting).
    """

    @abstractmethod
    def sample_error_bars(self, wlen: np.ndarray) -> np.ndarray:
        """
        Sample error bars for a spectrum.

        Args:
            wlen: Wavelengths (i.e., bin positions) of the spectrum.
                Expected shape: (n_bins, ).

        Returns:
            error_bars: Error bars for the spectrum. Shape: (n_bins, ).
        """

    @abstractmethod
    def sample_noise(self, error_bars: np.ndarray) -> np.ndarray:
        """
        Sample a noise realization from a Gaussian distribution with
        the given error bars. (Bins are assumed to be independent.)

        Args:
            error_bars: Error bars of the target spectrum.
                Expected shape: (n_bins, ).

        Returns:
            noise: Noise realization. Shape: (n_bins, ).
        """


class DefaultNoiseGenerator(NoiseGenerator):
    """
    Default noise generator that produces Gaussian noise with mean zero
    and a covariance matrix of the form `sigma^2 * I`, where `sigma` is
    sampled uniformly from the interval `[sigma_min, sigma_max]`.
    """

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        random_seed: int = 42,
    ) -> None:
        """
        Create a new instance of the default noise generator.

        Args:
            sigma_min: Minimum standard deviation of the noise.
            sigma_max: Maximum standard deviation of the noise.
            random_seed: Random seed for reproducibility.
        """

        # Store constructor arguments and initialize the RNG
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rng = np.random.default_rng(random_seed)

        # Check that the input parameters are valid
        if sigma_min < 0 or sigma_max < 0:
            raise ValueError("sigma values must be non-negative!")

    def sample_error_bars(self, wlen: np.ndarray) -> np.ndarray:
        """
        Sample error bars.
        """

        # Sample the standard deviation of the noise
        sigma = self.rng.uniform(self.sigma_min, self.sigma_max)

        # Error bars are the same for all bins
        error_bars = np.full(wlen.shape, sigma, dtype=np.float32)

        return error_bars

    def sample_noise(self, error_bars: np.ndarray) -> np.ndarray:
        """
        Sample a noise realization from a Gaussian distribution with
        the given `error_bars`. (Bins are assumed to be independent.)
        """

        # Draw noise with mean 0 and standard deviation equal to `error_bars`
        # noinspection PyTypeChecker
        return self.rng.normal(loc=0, scale=error_bars)


def get_noise_generator(config: dict) -> NoiseGenerator:
    """
    Create a noise generator based on the given `config`.

    Currently, this function is probably a bit overkill, but it may
    be useful if we want to add more noise generators in the future
    without changing the calling code.
    """

    # Get the noise generator type
    noise_generator_type = config["type"]
    noise_generator_kwargs = config["kwargs"]

    # Create the noise generator
    if noise_generator_type == "DefaultNoiseGenerator":
        return DefaultNoiseGenerator(**noise_generator_kwargs)
    else:
        raise ValueError(f"Unknown noise generator: {noise_generator_type}")
