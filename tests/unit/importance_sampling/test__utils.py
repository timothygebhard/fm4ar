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
    log_likelihoods = np.log(np.array([0, 1, 2]))
    log_prior_values = np.log(np.array([2, 4, 8]))
    log_probs = np.log(np.array([1e-1, 1e-2, 1e-3]))
    raw_log_weights, normalized_weights = compute_is_weights(
        log_likelihoods=log_likelihoods,
        log_prior_values=log_prior_values,
        log_probs=log_probs,
    )
    assert np.allclose(
        raw_log_weights,
        np.log(np.array([0, 4e2, 16e3])),
    )
    assert np.allclose(np.sum(normalized_weights), 3)
    assert np.allclose(normalized_weights, [0., 0.07317073, 2.92682927])
