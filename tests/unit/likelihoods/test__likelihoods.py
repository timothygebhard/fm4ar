"""
Unit tests for `fm4ar.likelihoods`.
"""

import typing

import numpy as np
import pytest
from pydantic import ValidationError

from fm4ar.likelihoods import get_likelihood_distribution
from fm4ar.likelihoods.config import LikelihoodConfig


@typing.no_type_check  # required because of the `noqa` comment
def test__likelihood_config() -> None:
    """
    Test `LikelihoodConfig`.
    """

    # Case 1: Valid config
    config = LikelihoodConfig(sigma=123)
    assert config.sigma == 123

    # Case 2: Invalid config
    with pytest.raises(ValidationError):
        LikelihoodConfig(invalid="unknown")  # noqa


def test__get_likelihood_distribution() -> None:
    """
    Test `get_likelihood_distribution()`.
    """

    # Case 1
    config = LikelihoodConfig(sigma=1)
    likelihood = get_likelihood_distribution(
        flux_obs=np.array([0., 0.]),
        config=config,
    )
    assert np.isclose(
        likelihood.logpdf(x=np.array([0., 0.])),
        np.log(1 / (2 * np.pi)),
    )

    # Case 2
    config = LikelihoodConfig(sigma=2)
    likelihood = get_likelihood_distribution(
        flux_obs=np.array([0., 0.]),
        config=config,
    )
    assert np.isclose(
        likelihood.logpdf(x=np.array([0., 0.])),
        np.log(1 / (8 * np.pi)),
    )
