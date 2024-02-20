"""
Methods for generating noise that can be added to the target spectrum.
"""

import re
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from fm4ar.utils.shapes import validate_dims


class NoiseGenerator:
    def __init__(
        self,
        random_seed: int = 42,
        complexity: int = 5,
        transform: Callable = lambda x, y: y,
    ):
        """
        Initialize the noise generator.

        For simplicity, we will assume the error bars are determined by
        some smooth random function over the wavelength range. The
        `complexity` of the random function can be adjusted to control
        the "smoothness" (i.e., the similarity of neighboring bins).

        By default, the error bars are all in [0, 1], but they can be
        transformed to any other range via the `transform` function.

        A special case is `complexity=0`, which will result in constant
        error bars. Combined with a suitable `transform`, this can be
        used to simulate a constant target noise level across all bins.

        Args:
            random_seed: Random seed (for reproducibility).
            complexity: Complexity of the random function. A higher
                complexity will result in less smooth error bars.
            transform: Transformation to apply to the random function,
                e.g., to shift or the scale the error bars.
        """

        self.rng = np.random.RandomState(random_seed)
        self.transform = transform

        if complexity < 0:
            raise ValueError("Complexity must be >= 0")
        self.complexity = complexity

    def sample_error_bars(
        self,
        wlen: np.ndarray,
    ) -> np.ndarray:
        """
        Sample error bars for a spectrum.

        Args:
            wlen: Wavelengths (i.e., bin positions) of the spectrum.
                Expected shape: (n_bins, ).

        Returns:
            error_bars: Error bars for the spectrum. Shape: (n_bins, ).
        """

        # Basic sanity check on input shape
        validate_dims(wlen, ndim=1)

        # If complexity is 0, return a error bar that is 1 everywhere
        if self.complexity == 0:
            error_bars = np.ones_like(wlen, dtype=np.float32)

        # If complexity is 1, return a constant random error bar
        elif self.complexity == 1:
            factor = self.rng.uniform(0, 1)
            error_bars = factor * np.ones_like(wlen, dtype=np.float32)

        # For complexity > 1, generate a random function and interpolate it
        else:
            coeff = self.rng.uniform(0, 1, self.complexity)
            grid = np.linspace(wlen.min(), wlen.max(), len(coeff))
            f = PchipInterpolator(grid, coeff)
            error_bars = f(wlen).astype(np.float32)

        # Transform the error bars; make sure they are non-negative
        error_bars = np.clip(self.transform(wlen, error_bars), 0, None)

        return error_bars

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

        # Basic sanity check on input shape
        validate_dims(error_bars, ndim=1)

        # Using multivariate_normal() is quite slow, because it assumes a full
        # covariance matrix, but since the bins are independent, we can sample
        # each bin independently, which is much faster (3 orders of magnitude).
        return self.rng.normal(loc=0, scale=error_bars).astype(np.float32)


def get_noise_transform_from_string(string: str) -> Callable:
    """
    Get the noise transform function from a string.

    Args:
        string: String representation of the transform function, e.g.,
            "lambda x, y: y + 0.1".

    Returns:
        transform: Transform function.
    """

    # Ensure we don't just call eval() on any string
    # We only allow strings of the form "lambda <x>, <y>: ...".
    if re.match(r"^lambda [a-zA-Z_]*, [a-zA-Z_]*: .*$", string):
        return eval(string)  # type: ignore

    raise ValueError(f"Invalid transform string: '{string}'")
