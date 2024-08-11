"""
Create P-P plots for each atmospheric parameter.

The quantiles `q(theta_gt)` of the ground truth parameters should follow
a uniform distribution over [0, 1] if the model is well-calibrated and
when the test set is drawn directly from the prior ("default" case). To
test this, we can draw their empirical CDF against the CDF of a uniform
distribution on [0, 1] in what is called a P-P plot. If the posteriors
are well-calibrated, the result should be a diagonal line.

The ECDF can be estimated by computing the fraction of test set
retrievals where `q(theta_gt) <= q` for `q` in [0, 1], although we
use the `ecdf()` function from `scipy.stats` for this purpose because
it is faster and comes with a way to estimate the confidence intervals.

For other test sets, this does not hold: Consider, for example, the
parameter `T_1`, which is poorly constrained from a spectrum and for
which the posterior will almost always be close to the prior. If we
draw `theta` not from the prior (but, e.g., a Gaussian distribution
around  `theta_0`), the ground truth values will not span the full prior
range. Consequently, their quantiles under the estimated posterior
(which is approximately the prior) will not be uniformly distributed.
"""

import argparse
import re
import time
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ecdf, ks_1samp, uniform
from yaml import safe_load

from fm4ar.utils.hdf import load_from_hdf
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path
from fm4ar.utils.plotting import set_font_family

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE P-P PLOTS\n", flush=True)

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

    # Load the quantiles of the ground truth values for the estimated posterior
    results = {}
    for result in config["results"]:
        results[result["label"]] = load_from_hdf(
            file_path=expand_path(result["file_path"]),
            keys=["quantiles_without_is"],
            idx=slice(0, result.get("n_retrievals", None)),
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

    # Loop over the atmospheric parameters and create the P-P plots
    for i, name in enumerate(NAMES):

        print(f"Creating P-P plot for {name}...", end=" ", flush=True)

        # Prepare the figure
        pad_inches = 0.01
        fig, ax = plt.subplots(
            figsize=(
                config["figsize"][0] / 2.54,
                config["figsize"][1] / 2.54,
            )
        )
        fig.subplots_adjust(**config["subplots_adjust"])

        # Loop over the different methods to include
        for result in results.values():

            # Unpack the results
            quantiles = result["quantiles_without_is"][:, i]

            # Compute KS test
            if config.get("add_p_values", False):
                ks_result = ks_1samp(
                    x=quantiles,
                    cdf=uniform(loc=0, scale=1).cdf,
                    alternative="two-sided",
                    method="exact",
                )

                # Add the p-value to the plot legend
                ax.plot(
                    [],
                    markersize=0,
                    ls="",
                    color=result["color"],
                    label=f"p={ks_result.pvalue:.2f}",
                )

            # Use scipy's `ecdf()` function instead of manually computing the
            # the fraction via `[np.mean(quantiles <= q) for q in q_values]`.
            # This is faster and comes with a way to estimate the confidence
            # intervals around the empirical CDF.
            empirical_cdf = ecdf(quantiles).cdf

            # Define target CDF: For a well-calibrated model, the quantiles
            # should follow a uniform distribution on [0, 1].
            uniform_cdf = uniform(loc=0, scale=1).cdf

            # Evaluate both CDFs to plot them against each other
            z = np.linspace(0, 1, 1000)
            x = uniform_cdf(z)
            y = empirical_cdf.evaluate(z)

            # Plot the CDFs against each other
            ax.step(
                x,  # uniform CDF
                y,  # ECDF of the quantiles
                lw=1,
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
                    z,
                    ci.low.evaluate(z),
                    ci.high.evaluate(z),
                    fc=result["color"],
                    ec="none",
                    alpha=0.3,
                )

        # Adjust the axis: labels, limits, ticks, ...
        ax.set_box_aspect(1)
        ax.axline((0, 0), slope=1, color="black", ls="--", lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r"Uniform CDF", fontsize=config["fontsize_labels"])
        ax.set_ylabel(
            r"ECDF of $Q(\theta_\mathrm{{gt}}^i)$",
            fontsize=config["fontsize_labels"],
        )
        ax.tick_params(
            axis="both",
            length=2,
            labelsize=config["fontsize_ticks"],
        )

        if config.get("add_p_values", False):
            ax.legend(
                frameon=False,
                loc="lower right",
                labelcolor='markerfacecolor',
                fontsize=config["fontsize_ticks"],
            )

        # Save the figure
        file_name = f"{i:02d}_" + re.sub(r"\W+", "-", str(name)) + ".pdf"
        plt.savefig(plots_dir / file_name)
        plt.close(fig)

        print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
