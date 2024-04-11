"""
Define an abstract base class for priors.
"""

from abc import ABC

import numpy as np
from scipy.stats import rv_continuous, rv_discrete


class BasePrior(ABC):
    """
    Common base class for all priors.
    """

    # Every prior must define a base distribution, e.g. `scipy.stats.uniform`
    distribution: rv_continuous | rv_discrete

    # Every prior must define the names (and labels) of the parameters
    names: tuple[str]
    labels: tuple[str]

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

    def evaluate(
        self,
        theta: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """
        Compute the prior probability of a given sample.
        Note: This assumes that the parameters are independent!
        """

        # The `mask` is used to evaluate if `len(theta) < len(self.names)`, as
        # it the case when we do not infer all parameters. See the `transform`
        # method for more details.
        if mask is not None:
            theta_full = np.zeros_like(mask, dtype=float)
            theta_full[mask] = theta
            return float(self.distribution.pdf(theta_full)[mask].prod())

        # Without a mask, we can simply compute the product of the PDFs
        else:
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
