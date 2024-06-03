"""
Utility functions for distributions (e.g., smoothed histograms).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_smoothed_histogram(
    bins: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray | None,
    sigma: float | None = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smoothed histogram of the given samples.

    Args:
        bins: Bins to use for the histogram.
        samples: Samples to use for the histogram.
        weights: Weights to use for the histogram. Can be None.
        sigma: Standard deviation of the Gaussian kernel to use
            for smoothing. If None, no smoothing is applied.

    Returns:
        bin_centers: Bin centers of the histogram.
        smoothed_hist: Smoothed histogram of the samples.
    """

    # Compute histogram
    hist, _ = np.histogram(
        a=samples,
        weights=weights,
        bins=bins,
        density=True,
    )

    # Determine bin centers
    bin_centers = np.array(0.5 * (bins[:-1] + bins[1:]))

    # Smooth histogram, if desired
    if sigma is not None:
        hist = gaussian_filter1d(hist, sigma=sigma)

    # Normalize histogram to sum to 1
    hist /= np.sum(hist)

    return bin_centers, hist
