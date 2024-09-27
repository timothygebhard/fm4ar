"""
Create 1D marginal plots that illustrate "posterior broadening".
"""

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from fm4ar.utils.distributions import compute_smoothed_histogram
from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path
from fm4ar.utils.plotting import set_font_family


def load_results(config: dict) -> dict:
    """
    Load all posterior samples and weights.
    """

    # TODO: Do we want to limit the number of samples to a given ESS?

    print("Loading results:", flush=True)

    # Example structure: results[0.1]["fmpe"]["samples"]
    results: dict[float, dict[str, dict[str, np.ndarray]]] = {}

    for sigma in config["results"]:
        results[sigma] = {}
        for method, file_path in config["results"][sigma].items():
            print(f"Loading {file_path}...", end=" ", flush=True)
            file_path = expand_env_variables_in_path(file_path)
            results[sigma][method] = load_from_hdf(
                file_path=file_path,
                keys=["samples", "weights"],
            )
            n_samples = len(results[sigma][method]["samples"])
            print(f"Done! ({n_samples:,} samples)", flush=True)

    print("", flush=True)

    return results


def get_parameters(config: dict) -> tuple[np.ndarray, ...]:
    """
    Get the names, labels, ranges, and true values for the parameters.
    """

    # Import labels, names, and ranges for the parameters
    if config["dataset"] == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import (
            LABELS,
            LOWER,
            NAMES,
            UPPER,
        )
    else:
        raise ValueError("Unknown dataset!")

    # Determine subset of parameters to plot plus their true values
    idx = []
    true_values = []
    for key, value in config["parameters"].items():
        idx.append(NAMES.index(key))
        true_values.append(value)

    # Return the names, labels, ranges, and true values
    return (
        np.array(idx),
        np.array(NAMES)[idx],
        np.array(LABELS)[idx],
        np.array(LOWER)[idx],
        np.array(UPPER)[idx],
        np.array(true_values)
    )


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE PLOTS OF 1D MARGINALS\n", flush=True)

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=Path,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Load plot configuration
    with open(args.config_file, "r") as file:
        config = safe_load(file)

    # Get the labels / ranges for the parameters, load posterior samples
    idx, names, labels, lower, upper, true_values = get_parameters(config)
    results = load_results(config)

    # Set the font family (globally), define colormap
    set_font_family(config.get("font_family"))
    cmap = plt.get_cmap(config.get("cmap", "viridis"))

    # Create the output directory
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Loop over the parameters and create the marginal plots
    print("Creating plots:", flush=True)
    for i in range(len(idx)):

        print(f"Creating plot for {names[i]}...", end=" ", flush=True)

        # Check if we want to put each noise level in a separate axis
        split_noiselevels = config.get("split_noiselevels", False)
        height_factor = 1.5 if split_noiselevels else 1

        # Prepare the figure
        pad_inches = 0.01
        fig, axes = plt.subplots(
            figsize=(
                config["figsize"][0] / 2.54,
                config["figsize"][1] / 2.54 * height_factor,
            ),
            nrows=1 if not split_noiselevels else len(results),
            sharex="all",
            sharey="all",
        )

        # Determine the bins for the current parameter
        bins = np.linspace(lower[i], upper[i], config["n_bins"])

        # Plot the posterior estimates
        for j, sigma in enumerate(results.keys()):
            for method in results[sigma]:

                # Select the axis for plotting
                ax = axes[j] if split_noiselevels else axes  # type: ignore

                # Compute the smoothed histogram
                bin_centers, smoothed_hist = compute_smoothed_histogram(
                    bins=bins,
                    samples=results[sigma][method]["samples"][:, idx[i]],
                    weights=results[sigma][method]["weights"],
                    sigma=config["smoothing"],  # this is the smoothing sigma!
                )

                # Plot the smoothed histogram
                ax.plot(
                    bin_centers,
                    smoothed_hist,
                    lw=config["linewidth"],
                    color=cmap((j + 1) / (len(results.keys()) + 1)),
                    ls=config["linestyles"][method],
                )

        # Adjust the axis: limits, ticks, ground truth, ...
        ax_list = [axes] if not split_noiselevels else axes
        for ax in ax_list:  # type: ignore
            ax.set_xlim(lower[i], upper[i])
            ax.set_ylim(0, None)
            ax.set_xticks(np.linspace(lower[i], upper[i], 5)[1:-1])
            ax.set_yticks([])
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.tick_params(
                axis="x",
                length=2,
                labelsize=config["fontsize_ticks"],
            )
            ax.axvline(x=float(true_values[i]), lw=0.5, ls="--", color="gray")

        # Replace special characters in the parameter name
        file_name = (
            f"{idx[i]:02d}_"
            + re.sub(r"\W+", "-", str(names[i]))
            + ".pdf"
        )

        # Save the figure
        plt.subplots_adjust(**config["subplots_adjust"])
        plt.savefig(plots_dir / file_name, dpi=300)
        plt.close(fig)

        print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:,.1f} seconds!\n")
