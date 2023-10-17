"""
Plot results of importance sampling runs.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer


def create_posterior_plot(
    samples: np.ndarray,
    weights: np.ndarray,
    parameter_mask: np.ndarray,
    names: list[str],
    ground_truth: np.ndarray,
    sample_efficiency: float,
    experiment_dir: Path,
) -> None:
    """
    Create a corner plot of the posterior.
    """

    # Suppress the custom warnings from ChainConsumer
    logging.basicConfig(level=logging.ERROR)

    # Setup new corner plot
    c = ChainConsumer()

    # First add the original samples without weights
    c.add_chain(
        chain=samples[:, parameter_mask],
        weights=None,
        parameters=names,
        name="Original posterior",
    )

    # Then add the samples with the importance sampling weights
    c.add_chain(
        chain=samples[:, parameter_mask],
        weights=weights,
        parameters=names,
        name=rf"IS posterior ($\epsilon$={100 * sample_efficiency:.1f}%)",
    )

    # Create the plot
    c.configure(sigmas=[0, 1, 2, 3], summary=False, cloud=True)
    c.plotter.plot(truth=ground_truth.tolist())

    # Save the plot as a PNG (PDFs can be 50+ MB for 16 parameters)
    file_path = experiment_dir / "importance_sampling_posterior.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
