"""
Integration tests for `fm4ar.nested_sampling.samplers`.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.nested_sampling.utils import create_posterior_plot


@pytest.mark.slow
@pytest.mark.parametrize(
    "library, sampler_kwargs, expected_mean",
    [
        ("nautilus", {"use_pool": False}, -0.011953013518526663),
        ("dynesty", {"sampling_mode": "standard"}, -0.01342119680650299),
        ("dynesty", {"sampling_mode": "dynamic"}, -0.010379853604532594),
        ("multinest", {}, 0.015017932514734234),
    ],
)
@pytest.mark.filterwarnings(r"ignore:(?s).*Found Intel OpenMP")
def test__samplers(
    library: str,
    sampler_kwargs: dict[str, Any],
    expected_mean: float,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """
    Test that we can run the different samplers.

    Note: This test (for multinest) seems to work only when running
    `pytest` directly, not when using PyCharm's test runner.
    """

    experiment_dir = tmp_path / library
    experiment_dir.mkdir()

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 10 * (u - 0.5)

    def log_likelihood(x: np.ndarray) -> float:
        return float(-0.5 * np.sum(x ** 2) - np.log(2 * np.pi))

    # Disable parallelization also for dynesty to ensure reproducibility
    if library == "dynesty":
        sampler_kwargs["use_pool"] = {
            "propose_point": True,
            "prior_transform": False,
            "loglikelihood": True,
        }

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

    # Check that we can compute the weighted average
    weighted_mean = sampler.get_weighted_posterior_mean()
    assert np.allclose(weighted_mean, np.zeros(2), atol=1e-1)

    # Check that we can plot the result
    create_posterior_plot(
        points=points,
        weights=weights,
        names=["x", "y"],
        file_path=experiment_dir / "posterior.pdf",
        ground_truth=np.array([0, 0]),
    )

    # Reset the captured output buffer; we only care about the output that is
    # produced after this point
    _ = capsys.readouterr()

    # Stop the test here for dynamic nested sampling with dynesty because
    # that one will just not stop sampling...
    if library == "dynesty" and sampler_kwargs["sampling_mode"] == "dynamic":
        return

    # Test what happens if we resume an already completed run
    # In this case, we should not see any output, but the sampler should
    # simply load the existing results
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
    sampler.run(max_runtime=60, verbose=True)
    sampler.cleanup()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert sampler.points is not None
    assert sampler.weights is not None
    assert np.allclose(points, loaded_points)
    assert np.allclose(weights, loaded_weights)
