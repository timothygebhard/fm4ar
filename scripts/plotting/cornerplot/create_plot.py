"""
Create a corner plot from a configuration file.
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from matplotlib.lines import Line2D
from tabulate import tabulate

from fm4ar.utils.config import load_config
from fm4ar.utils.distributions import compute_smoothed_histogram
from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file.",
    )
    return parser.parse_args()


def load_results(config: dict) -> dict:
    """
    Load all the posterior samples and weights.
    """

    results = {}
    for result in config["results"]:

        # Define shortcuts
        file_path = expand_env_variables_in_path(result["file_path"])
        label = result["label"]
        use_weights = result["use_weights"]

        print(f"Loading {file_path}...", end=" ", flush=True)

        # Load the samples and weights
        results[label] = load_from_hdf(file_path)
        if not use_weights:
            results[label]["weights"] = np.ones_like(results[label]["weights"])
            results[label]["sampling_efficiency"] = np.array(np.nan)
            results[label]["log_evidence"] = np.array(np.nan)
            results[label]["log_evidence_std"] = np.array(np.nan)

        # FIXME: Do we need to limit the number of samples to a given ESS?

        # Add the label and color
        results[label]["label"] = result["label"]
        results[label]["color"] = result["color"]

        # Construct extra dict with information of ESS, log evidence, etc.
        sampling_efficiency = results[label].get("sampling_efficiency", np.nan)
        n_total = len(results[label]["samples"])
        n_eff = n_total * sampling_efficiency
        log_Z = (
            f"{results[label]['log_evidence']:.4f} +/- "
            f"{results[label]['log_evidence_std']:.4f}"
        )
        results[label]["info"] = {  # type: ignore
            "n_total": n_total,
            "n_eff": n_eff,
            "sampling_efficiency": float(sampling_efficiency),
            "log_Z": log_Z,
        }

        print(f"Done! ({n_total:,} samples)", flush=True)

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
    print("\nCREATE CORNER PLOT\n", flush=True)

    # Get command line arguments and load the configuration
    args = get_cli_arguments()
    file_dir, file_name = args.config.parent, args.config.name
    config = load_config(experiment_dir=file_dir, file_name=file_name)

    # Define shortcuts
    lw = config["linewidth"]

    # Load the posterior samples and weights
    print("Loading results:", flush=True)
    results = load_results(config)
    print("", flush=True)

    # Get the labels and ranges for the parameters
    idx, names, labels, lower, upper, true_values = get_parameters(config)
    table = tabulate(
        {
            "": idx,
            "Name": names,
            "Label": labels,
            "Lower": lower,
            "Upper": upper,
            "True": true_values,
        },
        headers="keys",
    )
    print(f"Parameters:\n\n{table}\n\n", flush=True)

    # Set up a different default font
    font = 'Gillius ADF'
    plt.rcParams['font.sans-serif'] = font
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = f'{font}:italic'
    plt.rcParams['mathtext.bf'] = f'{font}:bold'
    plt.rcParams['mathtext.cal'] = f'{font}:italic'

    # Prepare the figure
    print("Preparing figure...", end=" ", flush=True)
    N = len(names)
    pad_inches = 0.01
    fig = plt.figure(
        figsize=(
            config["figsize"][0] / 2.54 - 2 * pad_inches,
            config["figsize"][1] / 2.54 - 2 * pad_inches,
        ),
    )
    print("Done!", flush=True)

    # Plot posterior estimates
    print("Creating corner plot...", end=" ", flush=True)
    for _, result in results.items():
        fig = corner(
            fig=fig,
            data=result["samples"][:, idx],
            weights=result["weights"],
            color=result["color"],
            bins=config["n_bins"],
            range=[(a, b) for a, b in zip(lower, upper, strict=True)],
            smooth=config["smoothing"],
            smooth1d=config["smoothing"],
            plot_datapoints=False,
            plot_contours=True,
            plot_density=False,
            levels=(0.68, 0.95),
            contour_kwargs=dict(linewidths=lw),
        )
    print("Done!", flush=True)

    # Extract the axes
    axes = np.array(fig.axes).reshape((N, N))

    # Fix the histograms on the diagonal
    print("Fixing diagonals...", end=" ", flush=True)
    for i in range(N):

        # Select ax and clear existing plot
        ax = axes[i, i]
        ax.clear()
        ax.set_yticks([])

        # Determine bins for the current parameter
        bins = np.linspace(lower[i] , upper[i], config["n_bins"])

        # Compute and plot histograms for all results
        for _, result in results.items():
            bin_centers, smoothed_hist = compute_smoothed_histogram(
                bins=bins,
                samples=result["samples"][:, idx[i]],
                weights=result["weights"],
                sigma=config["smoothing"],
            )
            ax.plot(
                bin_centers,
                smoothed_hist,
                linewidth=lw,
                color=result["color"]
            )

    print("Done!", flush=True)
    print("Adding ground truth values...", end=" ", flush=True)

    for row, col in product(range(N), range(N)):
        ax = axes[row, col]
        if col > row:
            continue
        elif col == row:
            ax.axvline(x=true_values[col], lw=lw / 1.5, ls="--", color="gray")
        else:
            ax.axhline(y=true_values[row], lw=lw / 1.5, ls="--", color="gray")
            ax.axvline(x=true_values[col], lw=lw / 1.5, ls="--", color="gray")

    print("Done!", flush=True)
    print("Fixing limits, ticks, and font sizes...", end=" ", flush=True)

    # Fix the ax limits
    for row, col in product(range(N), range(N)):
        ax = axes[row, col]
        if col > row:
            continue
        ax.set_xlim(lower[col], upper[col])
        if row == col:
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(lower[row], upper[row])

    # Adjust / remove ticks
    for row, col in product(range(N), range(N)):
        ax = axes[row, col]
        if col == 0 and row > 0:
            yticks = np.linspace(lower[row], upper[row], 5)[1:-1]
            yticks = np.around(yticks, 2)
            ax.set_yticks(yticks)
        else:
            ax.set_yticks([])
        if row < N - 1:
            ax.set_xticks([])
        else:
            xticks = np.linspace(lower[col], upper[col], 5)[1:-1]
            xticks = np.around(xticks, 2)
            ax.set_xticks(xticks)

    # Fix the labels
    # This is needed align the labels correctly instead of having their
    # position be determined by the longest tick label
    for row in range(1, N):
        ax = axes[row, 0]
        ax.text(
            x=config["label_offset_x"],
            y=0.5,
            s=labels[row],
            fontsize=config["fontsize_labels"],
            va="center",
            ha="right",
            rotation=90,
            transform=ax.transAxes
        )
    for col in range(N):
        ax = axes[N - 1, col]
        ax.text(
            x=0.5,
            y=config["label_offset_y"],
            s=labels[col],
            fontsize=config["fontsize_labels"],
            va="top",
            ha="center",
            transform=ax.transAxes,
        )

    # Fix the length and the fontsize of the ticks
    for ax, axis in product(axes.flatten(), ("x", "y")):
        ax.tick_params(
            axis=axis,
            length=2,
            labelsize=config["fontsize_ticks"],
            labelrotation=45 if axis == "x" else 0,
        )

    print("Done!", flush=True)
    print("Adding custom legend...", end=" ", flush=True)

    # Add a custom legend
    handles = [Line2D([0], [0], lw=lw / 1.5, ls="--", color="gray")]
    handles += [
        Line2D([0], [0], color=c, lw=4)
        for c in [result["color"] for result in results.values()]
    ]
    fig.legend(
        handles=handles,
        labels=["Ground truth"] + [r["label"] for r in results.values()],
        ncols=1,
        frameon=False,
        loc="upper right",
        fontsize=config["fontsize_labels"],
        bbox_to_anchor=(0.95, 0.95),
    )

    print("Done!", flush=True)
    print("Saving figure to PDF...", end=" ", flush=True)

    # Prepare the output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Save the info dicts to a JSON file
    file_name = (output_dir / config["output_file_name"]).stem + "__info.json"
    with open(output_dir / file_name, "w") as f:
        json.dump(
            obj={k: v["info"] for k, v in results.items()},
            fp=f,
            sort_keys=True,
            indent=4,
        )

    # Save the figure to PDF
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)
    plt.savefig(
        fname=output_dir / config["output_file_name"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )

    print("Done!", flush=True)
    print(f"\nThis took {time.time() - script_start:,.1f} seconds!\n")
