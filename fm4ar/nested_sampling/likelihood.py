"""
Define methods for creating likelihood distributions.
"""

from typing import Protocol

import numpy as np
from scipy.stats import multivariate_normal

from fm4ar.nested_sampling.config import LikelihoodConfig


class LikelihoodDistribution(Protocol):
    """
    Protocol for a distribution.

    This is used purely for typehinting, that is, it can be used to
    indicate that a function returns an object that has a `logpdf`
    method.
    """

    def logpdf(self, x: np.ndarray) -> float:
        raise NotImplementedError


def get_likelihood_distribution(
    x_obs: np.ndarray,
    config: LikelihoodConfig,
) -> LikelihoodDistribution:

    # Construct the covariance matrix from the given configuration
    # TODO: We need to figure out a way to specify generic covariance matrices
    #   in the configuration file. For now, we just assume that the covariance
    #   matrix is given as `sigma * np.eye(len(x_obs))`.
    cov = config.sigma * np.eye(len(x_obs))

    return multivariate_normal(mean=x_obs, cov=cov)  # type: ignore
