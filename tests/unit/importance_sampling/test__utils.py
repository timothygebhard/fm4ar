"""
Tests for `fm4py.importance_sampling.utils`.
"""

import numpy as np

from fm4ar.importance_sampling.utils import (
    compute_effective_sample_size,
    compute_is_weights,
)


def test__compute_effective_sample_size() -> None:
    """
    Test `compute_effective_sample_size()`.
    """

    # Case 1
    weights = np.array([0, 1])
    n_eff, sampling_efficiency = compute_effective_sample_size(weights)
    assert np.isclose(n_eff, 1)
    assert np.isclose(sampling_efficiency, 0.5)

    # Case 2
    weights = np.array([1, 1])
    n_eff, sampling_efficiency = compute_effective_sample_size(weights)
    assert np.isclose(n_eff, 2)
    assert np.isclose(sampling_efficiency, 1)


def test__compute_is_weights() -> None:
    """
    Test `compute_is_weights()`.
    """

    # Case 1
    log_likelihoods = np.array([-1, -2, -4])
    log_prior_values = np.array([-2, -4, -8])
    log_probs = np.array([-3, -6, -12])
    raw_log_weights, normalized_weights = compute_is_weights(
        log_likelihoods=log_likelihoods,
        log_prior_values=log_prior_values,
        log_probs=log_probs,
    )
    assert np.allclose(
        raw_log_weights,
        np.array([0, 0, 0]),
    )
    assert np.allclose(np.sum(normalized_weights), 3)
    assert np.allclose(normalized_weights, [1., 1., 1.])
