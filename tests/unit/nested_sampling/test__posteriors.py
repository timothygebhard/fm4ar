"""
Unit tests for `fm4ar.nested_sampling.posteriors`.
"""

from pathlib import Path

import pytest

from fm4ar.nested_sampling.posteriors import load_posterior


def test__load_posterior(tmp_path: Path) -> None:
    """
    Test `load_posterior()`.
    """

    # This tests only the trivial bits; the rest is tested in the tests of
    # the samplers themselves
    with pytest.raises(RuntimeError) as runtime_error:
        load_posterior(experiment_dir=tmp_path)
    assert "Could not determine the sampler type!" in str(runtime_error)