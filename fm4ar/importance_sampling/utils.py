"""
Utility functions for importance sampling.
"""

from warnings import warn

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

    # In case any weights are NaN of Inf, set them to 0
    raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the weights such that they sum to the number of samples.
    # If the sum of the raw weights is zero, issue a warning and set the
    # normalized weights to a uniform distribution.
    if np.sum(raw_weights) == 0:
        warn(UserWarning("All raw_weights are zero!"))
        normalized_weights = np.ones_like(raw_weights) / len(raw_weights)
    else:
        normalized_weights = (  # fmt: off
            raw_weights * len(raw_weights) / np.sum(raw_weights)
        )  # fmt: on

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
