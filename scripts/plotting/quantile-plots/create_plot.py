"""
Create quantile plots for each atmospheric parameter.

This is essentially the empirical CDF of the quantile of the ground
truth value for each parameter, that is, as a function of `q in [0, 1]`
the fraction of test set retrievals where `q(theta_gt) <= q`, where
`q(theta_gt)` is the quantile of the ground truth parameter value.

Note: In the current implementation, quantile plots can only be created
for the `default` test set (which is drawn directly from the box-uniform
prior). Generating the same plots for the `gaussian` test set would
require additional adjustments to account for the modified "prior".
"""

import argparse
import re
import time
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ecdf
from yaml import safe_load

from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path
from fm4ar.utils.plotting import set_font_family

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE QUANTILE PLOTS\n", flush=True)

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

    # Load the quantile data
    results = {}
    for result in config["results"]:
        results[result["label"]] = load_from_hdf(
            file_path=expand_path(result["file_path"]),
            keys=["quantiles_without_is", "quantiles_with_is"],
        )
        results[result["label"]] |= result

    # Set the font family (globally)
    set_font_family(config.get("font_family"))

    # Import parameter names
    if config["dataset"] == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import NAMES
    else:
        raise ValueError("Unknown dataset!")

    # Ensure the output directory exists
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Loop over the atmospheric parameters and create the quantile plots
    for i, name in enumerate(NAMES):

        print(f"Creating plot for {name}...", end=" ", flush=True)

        # Prepare the figure
        pad_inches = 0.01
        fig, ax = plt.subplots(
            figsize=(
                config["figsize"][0] / 2.54 - 2 * pad_inches,
                config["figsize"][1] / 2.54 - 2 * pad_inches,
            )
        )

        # Loop over the different methods to include
        for result in results.values():

            # Unpack the results
            use_weights = config.get("use_weights", False)
            quantiles = sorted(
                result["quantiles_with_is"][:, i] if use_weights
                else result["quantiles_without_is"][:, i]
            )

            # Use scipy's `ecdf()` function instead of manually computing the
            # the fraction via `[np.mean(quantiles <= q) for q in q_values]`.
            # This is faster and comes with a way to estimate the confidence
            # intervals around the empirical CDF.
            empirical_cdf = ecdf(quantiles).cdf

            # Manually plot the ECDF: The built-in `plot()` function from
            # `result.cdf` uses some odd hack with a `delta` that results in
            # a plot that exceeds beyond 0 and 1 on the x-axis.
            q = np.array([0] + list(empirical_cdf.quantiles) + [1])
            y = empirical_cdf.evaluate(q)
            ax.step(
                q,  # quantile
                y,  # value of the ECDF
                lw=1,
                label=result["label"],
                color=result["color"],
                ls=result.get("ls", "-"),
                where="post",
            )

            # Compute and plot the 95% confidence intervals
            # We suppress the RuntimeWarning that the CI is undefined for
            # some values, which should have no impact on the plot
            with catch_warnings():
                filterwarnings("ignore", "The confidence interval is ")
                ci = empirical_cdf.confidence_interval(0.95)
                ax.fill_between(
                    q,
                    ci.low.evaluate(q),
                    ci.high.evaluate(q),
                    fc=result["color"],
                    ec="none",
                    alpha=0.25,
                )

        # Adjust the axis: labels, limits, ticks, ...
        ax.set_box_aspect(1)
        ax.axline((0, 0), slope=1, color="black", ls="--", lw=0.5)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r"Quantile $q$", fontsize=config["fontsize_labels"])
        ax.set_ylabel("ECDF", fontsize=config["fontsize_labels"])
        ax.tick_params(
            axis="both",
            length=2,
            labelsize=config["fontsize_ticks"],
        )

        # Add legend, if desired
        if config.get("add_legend", False):
            ax.legend(
                loc="lower right",
                fontsize=config["fontsize_labels"],
                frameon=False,
            )

        # Save the figure
        fig.tight_layout(pad=0)
        file_name = f"{i:02d}_" + re.sub(r"\W+", "-", str(name)) + ".pdf"
        plt.savefig(
            plots_dir / file_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )
        plt.close(fig)

        print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
