"""
Create the table with the Jensen-Shannon Divergence results.
"""

import time
from copy import copy
from typing import Any

import h5py
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter1d
from tabulate import tabulate

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, LABELS, NAMES
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.utils.paths import get_experiments_dir


def get_sampling_efficiency(weights: np.ndarray) -> float:
    n_eff = np.sum(weights) ** 2 / np.sum(weights**2)
    return float(n_eff / len(weights))


def clip_weights(weights: np.ndarray, percentile: float) -> np.ndarray:
    threshold = np.percentile(weights, percentile)
    return np.clip(weights, 0, threshold)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return np.array(weights * len(weights) / np.sum(weights))


if __name__ == "__main__":
    script_start = time.time()
    print("\nCREATE JENSEN-SHANNON DIVERGENCE TABLE\n")

    # Define some constants
    PERCENTILE = 100.0
    SMOOTHING_SIGMA = 5.0

    # Define paths
    experiments_dir = get_experiments_dir()
    base_dir = experiments_dir / "aaai-workshop"

    # Load the "ground truth"
    print("Loading the ground truth...", end=" ", flush=True)
    file_path = (
        experiments_dir
        / "npe"
        / "vasist-2023"
        / "bigger-4"
        / "importance_sampling"
        / "importance_sampling_results.hdf"
    )
    results: dict[str, dict[str, Any]] = {}
    with h5py.File(file_path, "r") as f:
        results["ground_truth"] = dict(
            samples=np.array(f["theta"]),
            weights=np.array(f["weights"]),
        )
    print("Done!")

    # Define the experiment directories
    results["npe"] = {
        "label": "NPE",
        "dir": experiments_dir / "npe" / "vasist-2023" / "bigger-1",
        "weights": None,
    }
    results["fmpe_unfiltered"] = {
        "label": "FMPE (unfiltered)",
        "dir": base_dir / "fmpe-unfiltered",
        "weights": None,
    }
    results["fmpe_filtered"] = {
        "label": "FMPE (filtered)",
        "dir": base_dir / "fmpe-filtered",
        "weights": None,
    }
    results["fmpe_filtered_glu"] = {
        "label": "FMPE (filtered + GLU)",
        "dir": base_dir / "fmpe-filtered-glu",
        "weights": None,
    }
    results["fmpe_unfiltered_glu"] = {
        "label": "FMPE (unfiltered + GLU)",
        "dir": base_dir / "fmpe-unfiltered-glu",
        "weights": None,
    }

    # Add the importance sampling results as extra methods
    for key in copy(list(results.keys())):
        if key == "ground_truth":
            continue
        results[f"{key}_is"] = {
            "label": f"{results[key]['label']}  + IS",
            "dir": results[key]["dir"],
        }

    # Load the results and apply weight clipping
    print("Loading ML results...", end=" ", flush=True)
    for key in results.keys():

        # Skip the ground truth
        if key == "ground_truth":
            continue

        # Load the results
        file_path = (
            results[key]["dir"]
            / "importance_sampling"
            / "importance_sampling_results.hdf"
        )
        with h5py.File(file_path, "r") as f:
            results[key]["samples"] = np.array(f["theta"])
            if "weights" in results[key].keys():
                continue
            results[key]["raw_weights"] = np.array(f["raw_weights"])
            results[key]["weights"] = np.array(f["weights"])

        # Print the sampling efficiency (before clipping)
        epsilon = get_sampling_efficiency(results[key]["weights"])
        results[key]["epsilon_pre"] = epsilon

        # Clip the weights
        results[key]["weights"] = normalize_weights(
            clip_weights(
                weights=results[key]["raw_weights"],
                percentile=PERCENTILE,
            )
        )

        # Print the sampling efficiency (after clipping)
        epsilon = get_sampling_efficiency(results[key]["weights"])
        results[key]["epsilon_post"] = epsilon

    print("Done!")

    # Print the IS sampling efficiencies
    print("\nIS sampling efficiencies (pre / post clipping):")
    for key in results.keys():
        if "is" not in key:
            continue
        print(f"{results[key]['label']:<30}:", end=" ")
        print(f"{100 * results[key]['epsilon_pre']:.2f}%", end=" / ")
        print(f"{100 * results[key]['epsilon_post']:.2f}%")

    # Load nested sampling results
    print("\nLoading nested sampling results...", end=" ", flush=True)
    experiment_dir = (
        get_experiments_dir()
        / ".."
        / "scripts"
        / "nested_sampling"
        / "results"
        / "nautilus"
        # / "15_high-res"
        / "04-brilliant-flamingo"
    )
    samples, weights = load_posterior(experiment_dir=experiment_dir)
    results["nautilus"] = dict(
        label="nautilus",
        samples=samples,
        weights=weights,
    )
    print("Done!")

    # Compute histograms
    print("Computing histograms...", end=" ", flush=True)
    for key in results.keys():
        # Prepare the results dict
        results[key]["histograms"] = {}

        # Compute the smoothed histograms for each parameter
        for i, (lower, upper, name) in enumerate(zip(LOWER, UPPER, NAMES)):
            # Compute the raw histogram
            bins = np.linspace(lower, upper, 100)
            hist, _ = np.histogram(
                a=results[key]["samples"][:, i],
                bins=bins,
                weights=results[key]["weights"],
                density=True,
            )

            # Smooth the histogram
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            smoothed_hist = gaussian_filter1d(hist, sigma=SMOOTHING_SIGMA)

            results[key]["histograms"][name] = smoothed_hist

    print("Done!")

    # Compute JSD w.r.t. the ground truth
    print("Computing JSDs...", end=" ", flush=True)
    for key in results.keys():
        # Skip the ground truth
        if key == "ground_truth":
            continue

        # Compute the JSDs (in millinat)
        results[key]["jsds"] = {}
        for name in NAMES:
            p = results[key]["histograms"][name]
            q = results["ground_truth"]["histograms"][name]
            results[key]["jsds"][name] = 1000 * jensenshannon(p, q)

    print("Done!")

    # Prepare the results table
    print("\nJSD table:\n")
    table = []
    for method in results.keys():
        # Skip the ground truth
        if method == "ground_truth":
            continue

        # Add the label for the row
        row = [results[method]["label"]]

        # Add the JSD for each parameter to the row
        for name in NAMES:
            row.append(f"{results[method]['jsds'][name]:.2f}")

        # Add the mean JSD to the row
        vals = [results[method]["jsds"][name] for name in NAMES]
        row.append(f"{np.mean(vals):.2f}")

        # Add the row to the table
        table.append(row)

    # Sort the table by method name
    table = sorted(table, key=lambda row: row[0], reverse=True)

    # Print the table
    headers = ["Parameter", *LABELS, "Mean"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print()

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
