"""
Unit tests for `fm4ar.utils.distributions`.
"""

import numpy as np

from fm4ar.utils.distributions import compute_smoothed_histogram


def test__compute_smoothed_histogram() -> None:
    """
    Test `fm4ar.utils.distributions.compute_smoothed_histogram`.
    """

    rng = np.random.default_rng(42)

    # Case 1: No smoothing
    bins = np.linspace(0, 1, 10)
    samples = rng.random(100)
    weights = None
    bin_centers, hist = compute_smoothed_histogram(
        bins=bins,
        samples=samples,
        weights=weights,
        sigma=None,
    )
    assert len(bin_centers) == 9
    assert len(hist) == 9
    assert np.allclose(np.sum(hist), 1)

    # Case 2: Smoothing
    bins = np.linspace(0, 1, 10)
    samples = rng.random(100)
    weights = None
    bin_centers, hist = compute_smoothed_histogram(
        bins=bins,
        samples=samples,
        weights=weights,
        sigma=3,
    )
    assert len(bin_centers) == 9
    assert len(hist) == 9
    assert np.allclose(np.sum(hist), 1)
