"""
Methods for constructing likelihood distributions from configurations.
As of now, we only support multivariate normal distributions.
"""

from typing import Protocol

import numpy as np
from scipy.stats import multivariate_normal


class LikelihoodDistribution(Protocol):
    """
    Protocol for a distribution.

    This is used purely for typehinting, that is, it can be used to
    indicate that a function returns an object that has a `pdf` and
    a `logpdf` method.
    """

    def pdf(self, x: np.ndarray) -> float:
        raise NotImplementedError  # pragma: no cover

    def logpdf(self, x: np.ndarray) -> float:
        raise NotImplementedError  # pragma: no cover


def get_likelihood_distribution(
    flux_obs: np.ndarray,
    error_bars: np.ndarray,
) -> LikelihoodDistribution:

    # Construct the covariance matrix
    # For now, we simply assume that the errors are uncorrelated and that the
    # `error_bars` are the standard deviations of the Gaussian noise.
    cov = np.diag(error_bars ** 2)

    return multivariate_normal(mean=flux_obs, cov=cov)  # type: ignore
