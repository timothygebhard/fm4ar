"""
Utility functions for importance sampling.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from fm4ar.datasets.vasist_2023.prior import THETA_0
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.utils.hdf import load_from_hdf, save_to_hdf
from fm4ar.utils.paths import get_datasets_dir


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


def get_target_spectrum(
    args: Namespace,
    output_dir: Path,
    simulator: Simulator | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    # First, check if we already have the target spectrum
    if (output_dir / "target_spectrum.hdf").exists():
        target_spectrum = load_from_hdf(
            file_path=output_dir / "target_spectrum.hdf",
            keys=["wlen", "x_0"],
        )
        return target_spectrum["wlen"], target_spectrum["x_0"]

    # Then, check if we can load it from the test set
    if args.target_spectrum != "benchmark":

        # TODO: This part needs to be generalized for other datasets
        # TODO: Maybe pass the path explicitly?
        file_path = get_datasets_dir() / "vasist_2023" / "test" / "merged.hdf"
        target_spectrum = load_from_hdf(
            file_path=file_path,
            keys=["wlen", "x_0"],
            idx=args.target_spectrum,
        )

        # Save the target spectrum
        save_to_hdf(
            file_path=output_dir / "target_spectrum.hdf",
            **target_spectrum,
        )

        return target_spectrum["wlen"], target_spectrum["x_0"]

    # Otherwise, we need to simulate the target spectrum
    if simulator is None:
        raise RuntimeError("No target spectrum found, must provide simulator!")
    else:
        if (result := simulator(THETA_0)) is None:
            raise RuntimeError("Simulation of target spectrum failed!")
        else:
            return result[0], result[1]


def construct_context(
    x_0: np.ndarray,
    wlen: np.ndarray,
    SIGMA: float,
) -> torch.Tensor:

    # Construct uncertainties
    # TODO: This should be generalized for other datasets
    noise_level = SIGMA * np.ones_like(x_0)

    # Construct context
    context = (
        torch.stack(
            [
                torch.from_numpy(x_0),
                torch.from_numpy(wlen),
                torch.from_numpy(noise_level),
            ],
            dim=1,
        )
        .float()
        .unsqueeze(0)  # Add batch dimension
    )

    return context
