"""
Abstract base classes for priors and simulators of datasets.
"""

from abc import ABC, abstractmethod
from scipy.stats import rv_continuous, rv_discrete

import numpy as np


class BasePrior(ABC):
    """
    Common base class for all priors.
    """

    # Every prior must define a base distribution, e.g. `scipy.stats.uniform`
    distribution: rv_continuous | rv_discrete

    # Every prior must define the names of the parameters
    names: tuple[str]

    # Every prior must also define a random state for reproducibility
    random_state: np.random.Generator

    def __init__(self, random_seed: int) -> None:
        """
        Initialize the class instance.

        Args:
            random_seed: Random seed to use for reproducibility.
        """

        # Use random seed to initialize the random number generator
        self.random_state = np.random.default_rng(random_seed)

    def evaluate(self, theta: np.ndarray) -> float:
        """
        Compute the prior probability of a given sample.
        """

        return float(self.distribution.pdf(theta).prod())

    def sample(self) -> np.ndarray:
        """
        Draw a sample from the prior.
        """

        return np.asarray(
            self.distribution.rvs(random_state=self.random_state)
        )

    def transform(
        self,
        u: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform a sample from the unit cube to the prior space.
        """

        # The `mask` is used to transform if `len(u) < len(self.names)`, as it
        # the case when we do not infer all parameters. In this case, we need
        # to increase the length of `u` to match the length of `self.names`,
        # fill the rest with zeros, apply the transformation, and then return
        # only the values that correspond to the inferred parameters.
        if mask is not None:
            u_full = np.zeros_like(mask, dtype=float)
            u_full[mask] = u
            return np.asarray(self.distribution.ppf(u_full)[mask])

        # Without a mask, we can simply apply the transformation directly
        else:
            return np.asarray(self.distribution.ppf(u))


class BaseSimulator(ABC):
    """
    Common base class for all simulators.
    """

    @abstractmethod
    def __call__(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Run the simulation and return the result.

        Every simulator should take an array of parameters `theta`
        as its input, and return a tuple (wavelengths, fluxes) as
        its result (or None, in case the simulation failed).
        """

        raise NotImplementedError
