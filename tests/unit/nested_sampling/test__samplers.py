"""
Unit tests for `fm4ar.nested_sampling.samplers`.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.nested_sampling.utils import create_posterior_plot


@pytest.mark.slow
@pytest.mark.parametrize(
    "library, sampler_kwargs, expected_mean",
    [
        ("nautilus", {}, 0.011864517961803463),
        ("dynesty", {"sampling_mode": "standard"}, -0.01342119680650299),
        ("dynesty", {"sampling_mode": "dynamic"}, 0.009668962528773964),
        ("multinest", {}, 0.015017932514734234),
    ],
)
@pytest.mark.filterwarnings(r"ignore:(?s).*Found Intel OpenMP")
def test__samplers(
    library: str,
    sampler_kwargs: dict[str, Any],
    expected_mean: float,
    tmp_path: Path,
) -> None:
    """
    Test
    """

    experiment_dir = tmp_path / library
    experiment_dir.mkdir()

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 10 * (u - 0.5)

    def log_likelihood(x: np.ndarray) -> float:
        return float(multivariate_normal(np.zeros(2), np.eye(2)).logpdf(x))

    # Set up the sampler
    sampler = get_sampler(library)(
        run_dir=experiment_dir,
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=2,
        n_livepoints=100,
        inferred_parameters=["x", "y"],
        random_seed=42,
        **sampler_kwargs,
    )

    # Run the sampler and save the results
    sampler.run(
        max_runtime=60,
        verbose=True,
        run_kwargs={"maxcall": 5000} if library == "dynesty" else {},
    )
    sampler.cleanup()
    sampler.save_results()

    # Get the points and weights
    points = sampler.points
    weights = sampler.weights
    assert points is not None
    assert weights is not None

    # Check that the posterior mean is close to the expected mean
    assert np.isclose(np.mean(points), expected_mean)

    # Check that we can load the points
    loaded_points, loaded_weights = load_posterior(experiment_dir)
    assert np.allclose(points, loaded_points)
    assert np.allclose(weights, loaded_weights)

    # Check that we can plot the result
    create_posterior_plot(
        points=points,
        weights=weights,
        names=["x", "y"],
        file_path=experiment_dir / "posterior.pdf",
        ground_truth=np.array([0, 0]),
    )
