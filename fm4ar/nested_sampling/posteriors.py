"""
Utility functions for loading posteriors.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fm4ar.nested_sampling.config import load_config
from fm4ar.utils.misc import suppress_output


def load_posterior(experiment_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the posterior samples and weights from a directory.
    """

    # Load nested sampling configuration to determine sampler type
    try:
        config = load_config(experiment_dir)
    except FileNotFoundError as e:
        raise RuntimeError("Could not determine the sampler type!") from e

    # nautilus stores the posterior in a .npz file
    if config.sampler.library == "nautilus":
        file_path = experiment_dir / "posterior.npz"
        data = np.load(file_path)
        samples = data["points"]
        weights = np.exp(data["log_w"])

    # dynesty stores the posterior in a .pickle file
    elif config.sampler.library == "dynesty":
        file_path = experiment_dir / "posterior.pickle"
        results = pd.read_pickle(file_path)
        samples = np.array(results.samples)
        weights = np.array(results.importance_weights())

    # multinest stores the posterior in a .dat file
    elif config.sampler.library == "multinest":
        from pymultinest.analyse import Analyzer

        with suppress_output():
            analyzer = Analyzer(
                n_params=len(pd.read_json(experiment_dir / "params.json")),
                outputfiles_basename=(experiment_dir / "run").as_posix(),
            )
            samples = np.array(analyzer.get_equal_weighted_posterior()[:, :-1])
            weights = np.ones(len(samples))

    # ultranest stores the posterior in a .npz file
    elif config.sampler.library == "ultranest":
        file_path = experiment_dir / "posterior.npz"
        data = np.load(file_path)
        samples = data["points"]
        weights = data["weights"]

    # This should never happen; but the linter complains otherwise...
    else:  # pragma: no cover
        raise RuntimeError("Could not determine the sampler type!")

    return samples, weights
