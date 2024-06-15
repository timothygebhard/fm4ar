"""
Unit tests for `fm4ar.likelihoods`.
"""

import numpy as np

from fm4ar.likelihoods import get_likelihood_distribution


def test__get_likelihood_distribution() -> None:
    """
    Test `get_likelihood_distribution()`.
    """

    # Case 1
    likelihood = get_likelihood_distribution(
        flux_obs=np.array([0.0, 0.0]),
        error_bars=np.array([1.0, 1.0]),
    )
    assert np.isclose(
        likelihood.logpdf(x=np.array([0.0, 0.0])),
        np.log(1 / (2 * np.pi)),
    )

    # Case 2
    likelihood = get_likelihood_distribution(
        flux_obs=np.array([0.0, 0.0]),
        error_bars=np.array([2.0, 2.0]),
    )
    assert np.isclose(
        likelihood.logpdf(x=np.array([0.0, 0.0])),
        np.log(1 / (8 * np.pi)),
    )
