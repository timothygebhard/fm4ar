"""
Utility functions for loading posteriors.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fm4ar.nested_sampling.config import load_config
from fm4ar.utils.misc import suppress_output


# Define shorthand for more readable code
SAMPLER_TYPE = Literal["nautilus", "dynesty", "multinest", "ultranest"]


def load_posterior(
    experiment_dir: Path,
    sampler_type: SAMPLER_TYPE | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the posterior samples and weights from a directory.

    Args:
        experiment_dir: Path to the experiment directory.
        sampler_type: Type of the nested sampling library used. This is
            usually determined automatically from the config file, but
            can be provided explicitly if needed.
    """

    # If no sampler type is provided, try to load it from the config file
    # This is the default, but for backward compatibility reasons, we still
    # allow the user to provide the sampler type explicitly
    if sampler_type is None:
        try:
            config = load_config(experiment_dir)
        except FileNotFoundError as e:  # pragma: no cover
            raise RuntimeError("Could not determine the sampler type!") from e
        sampler_type = config.sampler.library

    # nautilus stores the posterior in a .npz file
    if sampler_type == "nautilus":
        file_path = experiment_dir / "posterior.npz"
        data = np.load(file_path)
        samples = data["points"]
        weights = np.exp(data["log_w"])

    # dynesty stores the posterior in a .pickle file
    elif sampler_type == "dynesty":
        file_path = experiment_dir / "posterior.pickle"
        results = pd.read_pickle(file_path)
        samples = np.array(results.samples)
        weights = np.array(results.importance_weights())

    # multinest stores the posterior in a .dat file
    elif sampler_type == "multinest":
        from pymultinest.analyse import Analyzer

        with suppress_output():
            analyzer = Analyzer(
                n_params=len(pd.read_json(experiment_dir / "params.json")),
                outputfiles_basename=(experiment_dir / "run").as_posix(),
            )
            samples = np.array(analyzer.get_equal_weighted_posterior()[:, :-1])
            weights = np.ones(len(samples))

    # ultranest stores the posterior in a .npz file
    elif sampler_type == "ultranest":
        file_path = experiment_dir / "posterior.npz"
        data = np.load(file_path)
        samples = data["points"]
        weights = data["weights"]

    # Invalid sampler type
    else:
        raise ValueError(f"Invalid `sampler_type`: {sampler_type}!")

    return samples, weights
