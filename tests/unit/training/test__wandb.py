"""
Tests for `fm4ar.training.wandb`.
"""

from pathlib import Path

from fm4ar.training.wandb import get_wandb_id


def test__get_wandb_id(tmp_path: Path) -> None:
    """
    Test `fm4ar.training.wandb.get_wandb_id()`.
    """

    # Case 1: No wandb_id file
    wandb_id = get_wandb_id(experiment_dir=tmp_path)
    assert isinstance(wandb_id, str)
    assert len(wandb_id) == 8

    # Case 2: wandb_id file exists
    assert get_wandb_id(experiment_dir=tmp_path) == wandb_id
