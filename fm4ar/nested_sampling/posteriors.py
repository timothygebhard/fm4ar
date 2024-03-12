"""
Utility functions for loading posteriors.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_posterior(experiment_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the posterior samples and weights from a directory.
    """

    # Get the sampler type from the files present in the directory
    if (experiment_dir / "checkpoint.hdf5").exists():
        sampler = "nautilus"
    elif (experiment_dir / "checkpoint.save").exists():
        sampler = "dynesty"
    elif (experiment_dir / "run.txt").exists():
        sampler = "multinest"
    else:
        raise RuntimeError("Could not determine the sampler type!")

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

        analyzer = Analyzer(
            n_params=len(pd.read_json(experiment_dir / "params.json")),
            outputfiles_basename=(experiment_dir / "run").as_posix(),
        )
        samples = np.array(analyzer.get_equal_weighted_posterior()[:, :-1])
        weights = np.ones(len(samples))

    # This should never happen; but the linter complains otherwise...
    else:  # pragma: no cover
        raise ValueError(f"Unknown sampler: {sampler}")

    return samples, weights
