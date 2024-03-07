"""
Unit tests for `fm4ar.utils.config`.
"""

from pathlib import Path

import pytest

from fm4ar.utils.config import load_config, save_config


def test__save_config__load_config(tmp_path: Path) -> None:
    """
    Test both ``save_config()`` and ``load_config()``.
    """

    # Case 1: We can't load a config that doesn't exist
    with pytest.raises(FileNotFoundError) as file_not_found_error:
        load_config(experiment_dir=tmp_path, file_name="test.yaml")
    assert "No test.yaml in" in str(file_not_found_error)

    # Case 2: We can save a config and load it again
    config = {"a": "a", "b": 2, "c": 3.141, "d": True, "e": [1, 2, 3]}
    save_config(config=config, experiment_dir=tmp_path, file_name="test.yaml")
    loaded_config = load_config(experiment_dir=tmp_path, file_name="test.yaml")
    assert config == loaded_config
