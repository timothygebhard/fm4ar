"""
Unit tests for `fm4ar.priors`.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from fm4ar.priors import get_prior
from fm4ar.priors.base import BasePrior
from fm4ar.priors.config import PriorConfig
from fm4ar.datasets.vasist_2023.prior import Prior as Vasist2023Prior


def test__prior_config() -> None:
    """
    Test `PriorConfig`.
    """

    # Case 1: Valid config
    config = PriorConfig(
        dataset="vasist_2023",
        parameters={"C/O": "infer"},
        random_seed=42,
    )
    assert config.dataset == "vasist_2023"
    assert config.parameters == {"C/O": "infer"}
    assert config.random_seed == 42

    # Case 2: Invalid config
    with pytest.raises(ValidationError):
        PriorConfig(dataset="unknown", parameters={})  # type: ignore


def test__get_prior() -> None:
    """
    Test `get_prior()`.
    """

    # -------------------------------------------------------------------------
    # Case 1: vasist_2023 prior
    # -------------------------------------------------------------------------

    config = PriorConfig(
        dataset="vasist_2023",
        parameters={},
        random_seed=42,
    )
    prior = get_prior(config=config)

    assert isinstance(prior, BasePrior)
    assert isinstance(prior, Vasist2023Prior)
    assert prior.sample().shape == (16, )

    assert np.isclose(
        prior.evaluate(prior.sample()),
        1.887564381187481e-09,
        atol=1e-10,
    )

    u = np.random.uniform(0, 1, 16)
    assert np.isclose(
        prior.evaluate(prior.transform(u=u, mask=None), mask=None),
        1.887564381187481e-09,
        atol=1e-10,
    )

    u = np.random.uniform(0, 1, 8)
    mask = np.array(8 * [0, 1], dtype=bool)
    assert np.isclose(
        prior.evaluate(prior.transform(u=u, mask=mask), mask=mask),
        0.004709095618186529,
        atol=1e-10,
    )
