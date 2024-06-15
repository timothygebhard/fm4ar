"""
Unit tests for `fm4ar.nested_sampling.samplers`.
"""

import json
import os
import time
from pathlib import Path
from shutil import copyfile
from typing import Any
from yaml import safe_dump

import numpy as np
import pytest

from fm4ar.nested_sampling.config import load_config
from fm4ar.nested_sampling.samplers import get_sampler
from fm4ar.utils.paths import get_experiments_dir


@pytest.mark.slow
@pytest.mark.parametrize(
    "library, sampler_kwargs, run_kwargs",
    [
        ("nautilus", {}, {}),
        ("dynesty", {}, {}),
        ("multinest", {}, {}),
        (
            "ultranest",
            {
                "stepsampler": {
                    "generate_direction": "generate_mixture_random_direction",
                    "nsteps": 10,
                }
            },
            {"n_calls_between_timeout_checks": 100},
        ),
        (
            "ultranest",
            {},
            {
                "n_calls_between_timeout_checks": 100,
                "region_class": "RobustEllipsoidRegion",
            },
        ),
    ],
)
@pytest.mark.filterwarnings(r"ignore:(?s).*Found Intel OpenMP")
def test__sampler_timeout(
    library: str,
    sampler_kwargs: dict[str, Any],
    run_kwargs: dict[str, Any],
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

    # Copy over the template configuration
    template_dir = get_experiments_dir() / "templates" / "nested-sampling"
    copyfile(
        template_dir / "config.yaml",
        experiment_dir / "config.yaml",
    )

    # Update the configuration
    config = load_config(experiment_dir)
    config.sampler.library = library  # type: ignore
    with open(experiment_dir / "config.yaml", "w") as yaml_file:
        safe_dump(
            json.loads(config.json()),
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
        )

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return 10 * (u - 0.5)

    def log_likelihood(x: np.ndarray) -> float:
        time.sleep(0.1)
        return float(-0.5 * np.sum(x ** 2))

    # Set up the sampler
    sampler = get_sampler(library)(
        run_dir=experiment_dir,
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        n_dim=2,
        n_livepoints=100,
        inferred_parameters=["x", "y"],
        random_seed=42,
        sampler_kwargs=sampler_kwargs,
    )

    # Run the sampler and save the results
    max_runtime = 10
    runtime = sampler.run(
        max_runtime=max_runtime,
        verbose=True,
        run_kwargs=run_kwargs,
    )
    sampler.cleanup()
    sampler.save_runtime(runtime=runtime)
    captured = capsys.readouterr()
    assert "stopping sampler!" in captured.out

    # The runtime limitation is not exact (especially for problems where
    # we are not dominated by the simulator), so we allow for some slack
    assert runtime < 1.5 * max_runtime
