"""
Unit tests for `fm4ar.nested_sampling.samplers`.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from fm4ar.nested_sampling.samplers import get_sampler


@pytest.mark.slow
@pytest.mark.parametrize(
    "library",
    [
        "nautilus",
        "dynesty",
        "multinest",
    ],
)
@pytest.mark.filterwarnings(r"ignore:(?s).*Found Intel OpenMP")
def test__sampler_timeout(
    library: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """
    Test the timeout functionality of the samplers.

    Note: This test (for multinest) seems to work only when running
    `pytest` directly, not when using PyCharm's test runner.
    """

    # Fail this test if it runs in PyCharm (it cannot handle multinest)
    if library == "multinest" and "PYCHARM_HOSTED" in os.environ:
        pytest.fail("This test does not work in PyCharm!")

    experiment_dir = tmp_path / library
    experiment_dir.mkdir()

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return u

    def log_likelihood(_: np.ndarray) -> float:
        time.sleep(2)
        return 10 + np.random.normal(0, 1)  # dynesty breaks if all are equal

    # Set up the sampler
    sampler = get_sampler(library)(
        run_dir=experiment_dir,
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=2,
        n_livepoints=10,
        inferred_parameters=["x", "y"],
        random_seed=42,
    )

    # Run the sampler and save the results
    sampler.run(max_runtime=1, verbose=True)
    sampler.cleanup()
    captured = capsys.readouterr()
    assert "Timeout reached, stopping sampler!" in captured.out
