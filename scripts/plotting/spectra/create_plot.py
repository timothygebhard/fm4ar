"""
Create plot of a spectrum (with error bands)
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from yaml import safe_load

from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path
from fm4ar.utils.plotting import adjust_lightness, set_font_family

if __name__ == "__main__":

    script_start = time.time()
    print("\nPLOT SPECTRUM WITH ERROR BANDS\n")

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

    # Load the target spectrum
    print("Loading target spectrum...", end=" ", flush=True)
    target_spectrum = load_target_spectrum(
        file_path=expand_path(Path(config["target_spectrum"]["file_path"])),
        index=config["target_spectrum"]["index"],
    )
    print("Done!", flush=True)

    # Set the font family (globally)
    set_font_family(config.get("font_family"))

    # Prepare the figure
    print("Creating plot...", end=" ", flush=True)
    pad_inches = 0.01
    fig, ax = plt.subplots(
        figsize=(
            config["figsize"][0] / 2.54 - 2 * pad_inches,
            config["figsize"][1] / 2.54 - 2 * pad_inches,
        ),
    )

    # Plot the error bars and collect handles for the legend
    handles = []
    sigmas = sorted(config.get("sigmas", []))
    for i, sigma in enumerate(sigmas):
        amount = i / (len(sigmas) - 1) * 0.8
        color = adjust_lightness(color=config["color"], amount=amount)
        ax.fill_between(
            target_spectrum["wlen"],
            target_spectrum["flux"] - sigma,
            target_spectrum["flux"] + sigma,
            step="pre",
            fc=color,
            ec="none",
            zorder=len(sigmas) - i,
        )
        patch = Patch(
            fc=color,
            label=rf"$\sigma = {sigma}\,\times\,${config['unit']}",
        )
        handles.append(patch)

    # Plot the spectrum
    ax.step(
        target_spectrum["wlen"],
        target_spectrum["flux"],
        color="k",
        lw=0.5,
        solid_joinstyle='miter',
        zorder=1000,
    )

    # Add the legend
    ax.legend(
        handles=handles,
        loc='best',
        frameon=False,
        handlelength=1,
        handleheight=1,
        fontsize=config["fontsize_ticks"],
    )

    # Ensure spines are on top of the plot
    for spine in ax.spines.values():
        spine.set_zorder(1000)

    # Set the axis labels
    ax.set_xlabel(
        "Wavelength (in µm)",
        fontsize=config["fontsize_labels"],
    )
    ax.set_ylabel(
        f"Planet flux (in {config['unit']})",
        fontsize=config["fontsize_labels"],
    )

    # Set the axis limits
    ax.set_xlim(min(target_spectrum["wlen"]), max(target_spectrum["wlen"]))
    ax.set_ylim(0, None)

    # Adjust the tick label sizes
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=config["fontsize_ticks"],
    )

    print("Done!", flush=True)

    # Save the figure
    print("Saving plot...", end=" ", flush=True)
    fig.tight_layout(pad=0)
    plt.savefig(
        Path(__file__).parent / config["output_file_name"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
