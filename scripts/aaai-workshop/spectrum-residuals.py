"""
Compute distribution of residuals for spectrum reconstruction.
"""

import argparse
import ast
import pickle
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.datasets.vasist_2023.prior import THETA_0
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.nested_sampling.posteriors import load_posterior


def weighted_percentile(
    data: np.ndarray,
    weights: np.ndarray,
    percs: np.ndarray,
) -> np.ndarray:
    ix = np.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(percs, cdf, data)


def compute_residuals(
    target: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray,
    percentiles: Sequence[float] = (68.27, 95.45, 99.73),
) -> np.ndarray:
    """
    Source: https://stackoverflow.com/a/61343915/4100721
    """

    n_bins = len(target)
    residuals = samples.reshape(-1, n_bins) - target.reshape(1, n_bins)

    # Prepare results array
    results = np.full((2 * len(percentiles), n_bins), np.nan)

    # Collect values at which to evaluate
    x = list()
    for p in percentiles:
        x.append(0 + (100 - p) / 2)
        x.append(100 - (100 - p) / 2)
    percs = np.array(x) / 100

    # Compute percentiles
    for i in tqdm(range(n_bins), ncols=80):
        results[:, i] = weighted_percentile(
            data=residuals[:, i],
            weights=weights,
            percs=percs,
        )

    return results


if __name__ == "__main__":
    script_start = time.time()
    print("\nCOMPUTE SPECTRUM RESIDUALS\n")

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--use-weights",
        type=ast.literal_eval,
        default=True,
        help="Whether to weight samples or not.",
    )
    args = parser.parse_args()

    # Simulate ground truth spectrum
    print("Simulating ground truth...", end=" ", flush=True)
    simulator = Simulator(noisy=False, R=1000)
    _, target = simulator(theta=THETA_0)  # type: ignore
    print("Done!")

    # Load the samples
    print("Loading samples...", end=" ", flush=True)
    file_path = (
        args.experiment_dir
        / "importance_sampling"
        / "importance_sampling_results.hdf"
    )
    if file_path.exists():
        with h5py.File(file_path, "r") as f:
            samples = np.array(f["x"])
            weights = np.array(f["weights"])
    else:
        samples, weights = load_posterior(experiment_dir=args.experiment_dir)
    print("Done!\n")

    # Overwrite weights, if needed
    if not args.use_weights:
        print("Disabling weights...", end=" ", flush=True)
        weights = np.ones_like(weights)
        print("Done!\n")

    # Compute the residuals
    print("Computing residuals:")
    percentiles = (68.27, 95.45, 99.73)
    results = compute_residuals(
        target=target,
        samples=samples,
        weights=weights,
        percentiles=percentiles,
    )
    print()

    # Split the results into a dictionary (which is self-documenting)
    output = {}
    for i, p in enumerate(percentiles):
        output[p] = results[2 * i : 2 * (i + 1)]

    # Store the final results as a pickle file
    print("Saving results to pickle file...", end=" ", flush=True)
    file_path = (
        args.experiment_dir
        / f"spectrum-residuals__weights-{args.use_weights}.pickle"
    )
    with open(file_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
