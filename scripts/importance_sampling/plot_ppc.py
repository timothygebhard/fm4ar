"""
Create posterior predictive check (PPC) plots for the trained model.
"""

import argparse
import colorsys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from fm4ar.importance_sampling.config import load_config
from fm4ar.utils.hdf import load_from_hdf


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir",
        type=Path,
        required=True,
        help="Path to the working directory with the results.",
    )
    parser.add_argument(
        "--use-importance-weights",
        action=argparse.BooleanOptionalAction,
        help="Use importance weights to compute the coverage intervals.",
    )
    args = parser.parse_args()

    return args


def get_weighted_quantile_interpolator(
    a: np.ndarray,
    weights: np.ndarray,
) -> interp1d:
    """
    Get the `q`-th quantile of the array `a` when weighted by `weights`.
    """

    # Sort the data
    idx = np.argsort(a)
    a = a[idx]
    weights = weights[idx]

    # Compute the weighted quantiles
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)

    return interp1d(
        x=weighted_quantiles,
        y=a,
        assume_sorted=True,
    )


def get_coverage_intervals(
    spectra: np.ndarray,
    coverages: Sequence[float],
    weights: np.ndarray,
) -> dict[float, dict[str, np.ndarray]]:
    """
    Get the desired coverage intervals for the given spectra.
    """

    # For each interval,
    results = {
        coverage: {
            "lower": np.zeros(spectra.shape[1]),
            "upper": np.zeros(spectra.shape[1]),
        }
        for coverage in coverages
    }

    # Compute the quantiles for each wavelength bin
    for i in tqdm(np.arange(spectra.shape[1]), ncols=80):

        # Get the quantile interpolator for this wavelength bin
        interpolator = get_weighted_quantile_interpolator(
            a=spectra[:, i],
            weights=weights,
        )

        # Compute the lower and upper value for each coverage level
        for c in coverages:
            results[c]["lower"][i] = interpolator((1 - c) / 2)
            results[c]["upper"][i] = interpolator(1 - (1 - c) / 2)

    return results


def adjust_luminosity(
    color: Any,
    amount: float = 0.5,
) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the
    given amount.
    """

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0.0, min(1.0, amount * c[1])), c[2])


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE POSTERIOR PREDICTICE CHECK PLOTS\n", flush=True)

    # Get command line arguments
    args = get_cli_arguments()

    # Load the target spectrum
    print("Loading target spectrum...", end=" ", flush=True)
    file_path = args.working_dir / "target_spectrum.hdf"
    if file_path.exists():
        target_spectrum = load_from_hdf(
            file_path=file_path,
            keys=["wlen", "flux", "noise"],
        )
    else:
        target_spectrum = np.load(file_path)
    wlen = target_spectrum["wlen"].flatten()
    flux = target_spectrum["flux"].flatten()
    n_bins = wlen.shape[0]
    print("Done!")

    # Load the nested sampling configuration
    print("Loading nested sampling configuration...", end=" ", flush=True)
    config = load_config(args.working_dir)
    print("Done!")

    # Load the full importance sampling results
    print("Loading importance sampling results...", end=" ", flush=True)
    file_path = args.working_dir / "importance_sampling_results.hdf"
    results = load_from_hdf(
        file_path=file_path,
        keys=["flux", "weights"],
    )
    print("Done!\n")

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(r"Wavelength $\lambda$ (µm)")
    ax.set_ylabel(r"Planet flux (in arbitrary units)")
    ax.set_xlim(wlen[0], wlen[-1])
    ax.set_ylim(0, 1.1 * np.max(flux))

    # Plot the 1-sigma error band around the noise-free target spectrum
    sigma = config.likelihood.sigma
    if "noise" in target_spectrum.keys():
        noise = target_spectrum["noise"].flatten()
    else:
        noise = np.zeros(n_bins)
    ax.fill_between(
        wlen,
        flux - noise - sigma * np.ones(n_bins),
        flux - noise + sigma * np.ones(n_bins),
        alpha=0.2,
        label=r"$1\,\sigma$ band",
        fc="black",
        ec="None",
        step="pre",
    )

    # Compute the coverage intervals
    print("Computing confidence intervals for each wavelength:", flush=True)
    coverage_intervals = get_coverage_intervals(
        spectra=results["flux"],
        coverages=[0.9973, 0.9545, 0.6827],  # order for plotting
        weights=(
            results["weights"] if args.use_importance_weights
            else np.ones_like(results["weights"])
        ),
    )
    print("")

    # Create the posterior predictive check plot
    print("Plotting confidence intervals...", end=" ", flush=True)
    for i, (alpha, values) in enumerate(coverage_intervals.items()):
        ax.fill_between(
            wlen,
            values["lower"],
            values["upper"],
            label=f"{100 * alpha:.1f}% interval",
            fc=adjust_luminosity("C1", amount=1.6 - 0.3 * i),
            ec="None",
            step="pre",
        )
    print("Done!")

    # Finally: Overplot the target spectrum
    print("Plotting target spectrum...", end=" ", flush=True)
    if "noise" in target_spectrum.keys():
        ax.step(
            wlen,
            flux - target_spectrum["noise"].flatten(),
            lw=0.5,
            color="black",
            label="Input (without noise)",
        )
        ax.step(
            wlen,
            flux,
            lw=0.5,
            color="C0",
            label="Input (with noise)",
        )
    else:
        ax.step(
            wlen,
            flux,
            lw=0.5,
            color="black",
            label="Input (without noise)",
        )
    print("Done!")

    # Add legend to plot
    ax.legend(loc="best", frameon=False)

    # Save the figure
    print("Saving figure...", end=" ", flush=True)
    file_path = args.working_dir / "ppc.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
