"""
This defined some useful functions for the nested sampling scripts.
"""

from typing import Any
from pathlib import Path
from warnings import warn

import corner
import matplotlib.pyplot as plt
import numpy as np

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, NAMES, LABELS
from fm4ar.datasets.vasist_2023.simulation import Simulator


def get_target_parameters_and_spectrum(
    resolution: int = 1000,
    time_limit: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the simulator to get the target parameters and spectrum.
    """

    # Turn off simulator noise for the ground truth
    simulator = Simulator(noisy=False, R=resolution, time_limit=time_limit)

    # Define theta_obs; see Vasist et al. (2023)
    theta = np.array(
        [
            0.55,  # C/0
            0.0,  # Fe/H
            -5.0,  # log_pquench
            -0.86,  # log_X_cb_Fe(c)
            -0.65,  # log_X_cb_MgSiO3(c)
            3.0,  # fsed
            8.5,  # log_kzz
            2.0,  # sigma_lnorm
            3.75,  # log_g
            1.0,  # R_pl
            1063.6,  # T_int
            0.26,  # T3
            0.29,  # T2
            0.32,  # T1
            1.39,  # alpha
            0.48,  # log_delta
        ]
    )

    # Run the simulator
    result = simulator(theta)
    if result is None:
        raise RuntimeError("Failed to simulate ground truth!")
    _, x = result

    return theta, x


def get_subsets(
    parameters: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the indices, lower and upper bounds, and labels for the
    selected `parameters`.
    """

    # Get labels and ranges for the selected parameters
    if parameters is None:
        parameters = NAMES

    idx = np.array([list(NAMES).index(name) for name in parameters])
    lower = np.array(LOWER)[idx]
    upper = np.array(UPPER)[idx]
    labels = np.array(LABELS)[idx]

    return idx, lower, upper, labels


def create_posterior_plot(
    points: np.ndarray | None,
    weights: np.ndarray | None,
    parameters: list[str] | None = None,
    file_path: Path | None = None,
    ground_truth: np.ndarray | None = None,
    **kwargs: Any,
) -> None:
    """
    Create a corner plot of the posterior.
    """

    # If we did not get any points, we can stop here
    if points is None:
        warn("\nNo points to plot!\n", stacklevel=2)
        return

    # Get labels and ranges for the selected parameters
    if parameters is None:
        parameters = NAMES
    idx = np.array([list(NAMES).index(name) for name in parameters])
    labels = np.array(LABELS)[idx]
    lower = np.array(LOWER)[idx]
    upper = np.array(UPPER)[idx]

    # corner() will fail if there are less than 2 dimensions
    if (ndim := len(parameters)) <= 1:
        warn("\nCannot plot posterior with < 2 dimensions!\n", stacklevel=2)
        return

    # Define default corner plot kwargs and update with user kwargs
    corner_kwargs = {
        "bins": 20,
        "labels": labels,
        "smooth": 0.9,
        "color": "royalblue",
        "plot_datapoints": False,
        "plot_density": False,
        "fill_contours": True,
        "levels": (0.68, 0.95, 0.997),
        "range": list(zip(lower, upper, strict=True)),
    }
    corner_kwargs.update(kwargs)

    # Create the corner plot
    fig, axes = plt.subplots(ndim, ndim, figsize=(2 * ndim, 2 * ndim))
    corner.corner(
        fig=fig,
        data=points,
        weights=weights,
        **corner_kwargs,
    )

    # Add the ground truth values
    if ground_truth is not None:
        corner.overplot_lines(fig, ground_truth[idx], c="k", ls="-")
        corner.overplot_points(fig, ground_truth[idx][None], marker="s", c="k")

    # Save the figure
    fig.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
