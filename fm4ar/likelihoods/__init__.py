"""
Methods for constructing likelihood distributions from configurations.
As of now, we only support multivariate normal distributions.
"""

from typing import Protocol

import numpy as np
from scipy.stats import multivariate_normal

from fm4ar.likelihoods.config import LikelihoodConfig


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
    config: LikelihoodConfig,
) -> LikelihoodDistribution:

    # Construct the covariance matrix from the given configuration
    # TODO: We need to figure out a way to specify generic covariance matrices
    #   in the configuration file. For now, we just assume that the covariance
    #   matrix is given as `sigma * np.eye(len(x_obs))`.
    cov = config.sigma * np.eye(len(flux_obs))

    return multivariate_normal(mean=flux_obs, cov=cov)  # type: ignore
