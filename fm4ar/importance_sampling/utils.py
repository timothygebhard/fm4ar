"""
Utility functions for importance sampling.
"""

import numpy as np


def compute_is_weights(
    likelihoods: np.ndarray,
    prior_values: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the importance sampling weights.

    Args:
        likelihoods: Likelihood values.
        prior_values: Prior values.
        probs: Probabilities under the proposal distribution.

    Returns:
        raw_weights: Raw importance sampling weights.
        normalized_weights: Normalized importance sampling weights.
    """

    # Compute the raw weights
    raw_weights = likelihoods * prior_values / probs

    # Normalize the weights
    normalized_weights = raw_weights * len(raw_weights) / np.sum(raw_weights)

    return raw_weights, normalized_weights


def compute_effective_sample_size(
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the effective sample size.

    Args:
        weights: Importance sampling weights.

    Returns:
        n_eff: Effective sample size.
        sampling_efficiency: Sampling efficiency.
    """

    n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
    sampling_efficiency = float(n_eff / len(weights))

    return n_eff, sampling_efficiency
