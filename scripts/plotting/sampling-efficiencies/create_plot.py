"""
Create scatter plot of sampling efficiency for FMPE / NPE.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path
from fm4ar.utils.plotting import set_font_family

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE SCATTER PLOT OF SAMPLING EFFIENCIES\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration file for the plot.",
    )
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Load the data
    print("Loading data...", end=" ", flush=True)
    results = {}
    for key, file_path in config["results"].items():
        results[key] = load_from_hdf(
            file_path=expand_path(file_path),
            keys=["sampling_efficiency"],
        )
    print("Done!", flush=True)

    # Set the font family (globally)
    set_font_family(config.get("font_family"))

    # Setup bins for marginal histograms
    bins = np.linspace(0, 100, 101)

    # Type hints for axes are not supported by matplotlib
    axes: Any

    # Prepare the figure
    print("Creating plot...", end=" ", flush=True)
    fig, axes = plt.subplots(
        figsize=(
            config["figsize"][0] / 2.54,
            config["figsize"][1] / 2.54,
        ),
        ncols=2,
        nrows=2,
        gridspec_kw=dict(
            width_ratios=(7, 1),
            height_ratios=(1, 7),
            hspace=0.02,
            wspace=0.02,
        ),
        sharex="col",
        sharey="row",
    )

    # Disable empty axis
    axes[0, 1].axis("off")

    # Add scatterplot
    axes[1, 0].set_xlim(-2, 102)
    axes[1, 0].set_ylim(-2, 102)
    axes[1, 0].axline((0, 0), slope=1, color="black", ls="--", lw=0.5)
    axes[1, 0].scatter(
        100 * results["fmpe"]["sampling_efficiency"],
        100 * results["npe"]["sampling_efficiency"],
        s=config["marker_size"],
        c=config["color"],
    )
    axes[1, 0].tick_params(
        axis="both",
        length=2,
        labelsize=config["fontsize_ticks"],
    )

    # Add axis labels
    # We cannot directly use `set_xlabel()` and `set_ylabel()` here because
    # that will mess with the aspect ratio due to the tick label sizes...
    axes[1, 0].text(
        x=50,
        y=-15,
        s="FMPE sampling efficiency (in %)",
        fontsize=config["fontsize_labels"],
        ha="center",
        va="top",
    )
    axes[1, 0].text(
        x=-15,
        y=50,
        s="NPE sampling efficiency (in %)",
        fontsize=config["fontsize_labels"],
        ha="right",
        va="center",
        rotation=90,
    )

    # Add marginal histograms at the top
    axes[0, 0].tick_params(axis="both", length=0)
    axes[0, 0].set_yticklabels([])
    axes[0, 0].spines[["left", "right", "top"]].set_visible(False)
    hist_fmpe, *_ = axes[0, 0].hist(
        100 * results["fmpe"]["sampling_efficiency"],
        bins=bins,
        color="k",
        histtype="bar"
    )

    # Add marginal histograms on the right
    axes[1, 1].tick_params(axis="both", length=0)
    axes[1, 1].set_xticklabels([])
    axes[1, 1].spines[["right", "top", "bottom"]].set_visible(False)
    hist_npe, *_ = axes[1, 1].hist(
        100 * results["npe"]["sampling_efficiency"],
        bins=bins,
        color="k",
        histtype="bar",
        orientation="horizontal"
    )

    # Ensure the histograms have the same scale to allow direct comparison
    max_count = max(max(hist_fmpe), max(hist_npe))
    axes[0, 0].set_ylim(0, max_count)
    axes[1, 1].set_xlim(0, max_count)

    print("Done!", flush=True)

    # Compute and print the correlation coefficient
    corr_coeff = np.corrcoef(
        results["fmpe"]["sampling_efficiency"],
        results["npe"]["sampling_efficiency"],
    )[0, 1]
    print(f"\nCorrelation coefficient: {corr_coeff:.3f}\n")

    # Save the figure
    print("Saving plot...", end=" ", flush=True)
    fig.subplots_adjust(**config["subplots_adjust"])
    fig.savefig(Path(__file__).parent / config["output_file_name"])
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
