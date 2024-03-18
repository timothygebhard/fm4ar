"""
Unit tests for `fm4ar.nested_sampling.config`.
"""

from fm4ar.nested_sampling.config import load_config
from fm4ar.utils.paths import get_experiments_dir


def test__load_config() -> None:
    """
    Test `load_config`.
    """

    # Load the template config
    experiment_dir = get_experiments_dir() / "templates" / "nested-sampling"
    config = load_config(experiment_dir)

    # Check that the config is loaded correctly
    assert config.sampler.library == "nautilus"
    assert config.likelihood.sigma == 0.125754
