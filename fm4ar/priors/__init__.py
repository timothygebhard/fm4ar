"""
Convenience function for loading priors from a config.
"""

from fm4ar.priors.base import BasePrior
from fm4ar.priors.config import PriorConfig


def get_prior(config: PriorConfig) -> BasePrior:
    """
    Load a prior distribution from a configuration object.
    """

    if config.dataset == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import Prior
        return Prior(random_seed=config.random_seed)

    # This should never happen, because the `config` object is validated
    raise ValueError("Unknown prior dataset!")  # pragma: no cover
