"""
Method for loading priors from a configuration object.
"""

from fm4ar.datasets.base_classes import BasePrior
from fm4ar.nested_sampling.config import PriorConfig


def get_prior(config: PriorConfig) -> BasePrior:
    """
    Load a prior distribution from a configuration object.
    """

    if config.dataset == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import Prior
        return Prior(random_seed=config.random_seed)

    raise ValueError(f"Unknown prior dataset: {config.dataset}")
