"""
Create scatter plot of sampling efficiency for FMPE / NPE.
"""

import argparse
import time
from pathlib import Path

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

    # Prepare the figure
    print("Creating plot...", end=" ", flush=True)
    pad_inches = 0.01
    fig, axes = plt.subplots(
        figsize=(
            config["figsize"][0] / 2.54 - 2 * pad_inches,
            config["figsize"][1] / 2.54 - 2 * pad_inches,
        ),
        ncols=2,
        nrows=2,
        width_ratios=(7, 1),
        height_ratios=(1, 7),
    )

    # Disable empty axis
    axes[0, 1].axis("off")

    # Add scatterplot
    axes[1, 0].set_box_aspect(1)
    axes[1, 0].set_xlim(-2, 102)
    axes[1, 0].set_ylim(-2, 102)
    axes[1, 0].axline((0, 0), slope=1, color="black", ls="--", lw=0.5)
    axes[1, 0].scatter(
        100 * results["fmpe"]["sampling_efficiency"],
        100 * results["npe"]["sampling_efficiency"],
        s=config["marker_size"],
        c=config["color"],
    )
    axes[1, 0].set_xlabel(
        "FMPE sampling efficiency (in %)",
        fontsize=config["fontsize_labels"],
    )
    axes[1, 0].set_ylabel(
        "NPE sampling efficiency (in %)",
        fontsize=config["fontsize_labels"],
    )
    axes[1, 0].tick_params(
        axis="both",
        length=2,
        labelsize=config["fontsize_ticks"],
    )

    # Add marginal histograms at the top
    axes[0, 0].axis("off")
    axes[0, 0].set_xlim(-2, 102)
    axes[0, 0].hist(
        100 * results["fmpe"]["sampling_efficiency"],
        bins=bins,
        color="k",
        histtype="bar"
    )

    # Add marginal histograms on the right
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(-2, 102)
    axes[1, 1].hist(
        100 * results["npe"]["sampling_efficiency"],
        bins=bins,
        color="k",
        histtype="bar",
        orientation="horizontal"
    )

    print("Done!", flush=True)

    # Save the figure
    print("Saving plot...", end=" ", flush=True)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(
        Path(__file__).parent / config["output_file_name"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
