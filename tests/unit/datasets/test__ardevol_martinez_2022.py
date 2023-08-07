"""
Test loading the Ardevol Martinez et al. (2022) dataset.
"""

import pytest

from fm4ar.datasets import load_dataset
from fm4ar.utils.paths import get_datasets_dir


AM_2023_DIR = get_datasets_dir() / "ardevol-martinez-2022"


@pytest.mark.skipif(
    condition=not (AM_2023_DIR / "training" / "trans_type1.npy").exists(),
    reason="Ardevol-Martinez-2022 training dataset is not available!"
)
@pytest.mark.parametrize("chemistry_model", [1, 2])
@pytest.mark.parametrize("instrument", ["NIRSPEC", "WFC3"])
def test__load_ardevol_martinez_2022_training_dataset(
    instrument: str,
    chemistry_model: str,
) -> None:
    """
    Test loading the Ardevol Martinez (2023) training dataset.
    """

    # Define configuration
    config = {
        "data": {
            "name": "ardevol-martinez-2022",
            "which": "training",
            "instrument": instrument,
            "type": f"type-{chemistry_model}",
        }
    }

    # Load the dataset
    dataset = load_dataset(config=config)

    # Basic checks
    assert len(dataset) > 70_000
    assert dataset.theta_dim == 10 if chemistry_model == 1 else 7
    assert dataset.context_dim == (25, ) if instrument == "WFC3" else (403, )


@pytest.mark.skipif(
    condition=not (AM_2023_DIR / "test" / "merged.hdf").exists(),
    reason="Ardevol-Martinez-2022 test dataset is not available!"
)
@pytest.mark.parametrize("chemistry_model", [1, 2])
@pytest.mark.parametrize("instrument", ["NIRSPEC", "WFC3"])
def test__load_ardevol_martinez_2022_test_dataset(
    instrument: str,
    chemistry_model: str,
) -> None:
    """
    Test loading the Ardevol Martinez (2023) test dataset.
    """

    # Define configuration
    config = {
        "data": {
            "name": "ardevol-martinez-2022",
            "which": "test",
            "instrument": instrument,
            "type": f"type-{chemistry_model}",
        }
    }
    dataset = load_dataset(config=config)

    # Basic checks
    assert len(dataset) == 1000
    assert dataset.theta_dim == 10 if chemistry_model == 1 else 7
    assert dataset.context_dim == (25, ) if instrument == "WFC3" else (403, )
