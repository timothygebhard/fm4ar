"""
Unit tests for `fm4ar.nested_sampling.posteriors`.
"""

from pathlib import Path

import pytest

from fm4ar.nested_sampling.posteriors import load_old_posterior


def test__load_old_posterior(tmp_path: Path) -> None:
    """
    Test `load_old_posterior()`.
    """

    # This tests only the trivial bits; the rest is tested in the tests of
    # the samplers themselves
    with pytest.raises(ValueError) as value_error:
        load_old_posterior(
            experiment_dir=tmp_path,
            sampler_type="invalid",  # type: ignore
        )
    assert "Invalid `sampler_type`: invalid" in str(value_error)
