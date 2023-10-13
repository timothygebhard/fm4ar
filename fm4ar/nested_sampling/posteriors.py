"""
Utility functions for loading posteriors.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fm4ar.nested_sampling.config import load_config


def load_posterior(experiment_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the posterior samples and weights from a directory.
    """

    # Get the sampler type
    config = load_config(experiment_dir)
    sampler = config.sampler.which

    # nautilus stores the posterior in a .npz file
    if sampler == "nautilus":
        file_path = experiment_dir / "posterior.npz"
        data = np.load(file_path)
        samples = data["points"]
        weights = np.exp(data["log_w"])

    # dynesty stores the posterior in a .pickle file
    elif sampler == "dynesty":
        file_path = experiment_dir / "posterior.pickle"
        data = pd.read_pickle(file_path)
        samples = np.array(data.samples)
        weights = np.exp(data.logwt)

    # multinest stores the posterior in a .dat file
    elif sampler == "multinest":
        from pymultinest.analyse import Analyzer
        n_params = sum(p.action == "infer" for p in config.parameters.values())
        outputfiles_basename = (experiment_dir / "run").as_posix()
        analyzer = Analyzer(
            n_params=n_params,
            outputfiles_basename=outputfiles_basename,
        )
        samples = np.array(analyzer.get_equal_weighted_posterior()[:, :-1])
        weights = np.ones(len(samples))

    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    return samples, weights
