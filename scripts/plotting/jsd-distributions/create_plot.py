"""
Create violin plots of the Jensen-Shannon divergences between the
marginal posterior distributions with and without importance sampling,
for both FMPE and NPE.
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
    print("\nCREATE VIOLIN PLOTS OF JSD DISTRIBUTIONS\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="gaussian.yaml",
        help="Path to the configuration file for the plot.",
    )
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Load the data
    print("Loading data...", end=" ", flush=True)
    results = {}
    for key, result in config["results"].items():
        results[key] = load_from_hdf(
            file_path=expand_path(result["file_path"]),
            keys=["jsd_with_without_is", "sampling_efficiency"],
        )
        results[key]["color"] = result["color"]
    print("Done!", flush=True)

    # Import parameter names
    if config["dataset"] == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import LABELS
    else:
        raise ValueError("Unknown dataset!")

    # Set the font family (globally)
    set_font_family(config.get("font_family"))

    # Type hints for axes are not supported by matplotlib
    axes: Any

    # Prepare the figure
    # We need to call `subplots_adjust` already here, because otherwise the
    # `delta` we compute below for shifting the violins will not be correct
    print("Creating plot...", end=" ", flush=True)
    fig, ax = plt.subplots(
        figsize=(
            config["figsize"][0] / 2.54,
            config["figsize"][1] / 2.54,
        ),
    )
    fig.subplots_adjust(**config["subplots_adjust"])

    # Set up ticks, labels, and limits
    ax.set_xticks(range(len(LABELS)))
    ax.set_xlim(-0.5, len(LABELS) - 0.5)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 310)
    ax.set_ylabel("JSD (in mnat)", fontsize=config["fontsize_labels"])
    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=config["fontsize_labels"],
    )
    ax.tick_params(axis="y", length=2, labelsize=config["fontsize_ticks"])
    ax.spines[['right', 'top']].set_visible(False)

    # Plot the violins
    for offset, method in [(-1, "fmpe"), (1, "npe")]:

        # Mask out the methods with very low sampling efficiency
        # For these examples, the IS-based posterior is not reliable
        mask = results[method]["sampling_efficiency"] >= 0.01

        # Compute the delta by which we need to shift the violins such that
        # the lines indicating the support of the distribution do not overlap:
        # This needs to be exactly half the line with, but in data coordinates.
        line_width = config.get("linewidth", 1)  # in points
        points_per_inch = 72
        length_in_inch = fig.bbox_inches.width * ax.get_position().width
        x_range = np.diff(ax.get_xlim())[0]
        delta = line_width / (points_per_inch * length_in_inch / x_range) / 2

        # Plot the distribution of JSD values and their median
        violin: Any = ax.violinplot(
            dataset=results[method]["jsd_with_without_is"][mask],
            positions=np.arange(16) + delta * offset,
            showmedians=True,
            side="low" if offset < 0 else "high",
        )

        # Set colors for the different parts of the violin plot
        color = str(results[method]["color"])
        for partname in ('cbars', 'cmedians'):
            violin[partname].set_edgecolor(color)
            violin[partname].set_linewidth(line_width)
        for partname in ('cmins', 'cmaxes'):
            violin[partname].set_linewidth(0)
        for vp in violin['bodies']:
            vp.set_facecolor(color)
            vp.set_edgecolor("none")

    print("Done!", flush=True)

    # Save the figure
    print("Saving plot...", end=" ", flush=True)
    fig.savefig(Path(__file__).parent / config["output_file_name"])
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
