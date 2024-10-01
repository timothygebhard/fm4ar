"""
Create plots that show how the sampling efficiency scales with the
number of spectra used for training the model.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path
from fm4ar.utils.plotting import set_font_family

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE PLOT OF MEDIAN SAMPLING EFF. OVER TRAINING SET SIZE\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to the configuration file for the plot.",
    )
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Load the data
    print("Loading data...", end=" ", flush=True)
    results: DefaultDict = defaultdict(dict)
    for train_set_size, file_paths in config["results"].items():
        for method, file_path in file_paths.items():

            # Load data from HDF
            data = load_from_hdf(
                file_path=expand_path(file_path),
                keys=["sampling_efficiency"],
            )

            # Truncate to target number of retrievals
            n = config["n_retrievals"]
            sampling_efficiencies = data["sampling_efficiency"][:n]
            if len(sampling_efficiencies) < n:
                print(
                    f"Warning: Only {len(sampling_efficiencies)} retrievals "
                    f"results found in {file_path}."
                )
            else:
                results[method][train_set_size] = sampling_efficiencies

    print("Done!", flush=True)

    # Set the font family (globally)
    set_font_family(config.get("font_family"))

    # Prepare the figure
    print("Creating plot...", end=" ", flush=True)
    fig, ax = plt.subplots(
        figsize=(
            config["figsize"][0] / 2.54,
            config["figsize"][1] / 2.54,
        ),
    )

    # Set up ticks, labels, and limits
    ax.set_xticks(range(len(results["fmpe"])))
    ax.set_xlim(-0.5, len(results["fmpe"]) - 0.5)
    ax.set_xticklabels(
        list(results["fmpe"].keys()),
        fontsize=config["fontsize_ticks"],
    )
    ax.set_xlabel(
        "Size of the training dataset",
        fontsize=config["fontsize_labels"],
    )
    ax.set_ylabel(
        "Median sampling efficiency (in %)",
        fontsize=config["fontsize_labels"],
    )
    ax.tick_params(axis="y", length=2, labelsize=config["fontsize_ticks"])
    ax.spines[['right', 'top']].set_visible(False)

    # Plot the data
    # We are not adding error bars here because it makes the plot pretty messy
    for method in results:
        median = np.array(
            [
                np.median(sampling_efficiencies)
                for sampling_efficiencies in results[method].values()
            ]
        )
        ax.plot(
            range(len(median)),
            100 * np.array(median),  # convert to percentage
            "o:",
            color=config["colors"][method],
        )
    print("Done!", flush=True)

    # Save the figure
    print("Saving plot...", end=" ", flush=True)
    fig.subplots_adjust(**config["subplots_adjust"])
    fig.savefig(Path(__file__).parent / config["output_file_name"])
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
