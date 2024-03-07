"""
Unit tests for `fm4ar.models.build_model`.
"""

import pytest

from fm4ar.models.build_model import build_model


def test__build_model() -> None:
    """
    Test `build_model`.
    """

    # Case 1: Cannot call without arguments
    with pytest.raises(ValueError) as value_error:
        build_model()
    assert "Either `file_path` or `config` must be" in str(value_error)

    # Case 2: Cannot call with illegal model type
    with pytest.raises(ValueError) as value_error:
        build_model(config={"model": {"model_type": "invalid"}})
    assert "is not a valid model type!" in str(value_error)

    # All non-trivial use cases are tested in the integration tests!
