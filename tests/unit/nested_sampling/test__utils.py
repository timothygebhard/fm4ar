"""
Unit tests for `fm4ar.nested_sampling.utils`.
"""

import numpy as np
import pytest

from fm4ar.datasets.vasist_2023.prior import Prior as Vasist2023Prior
from fm4ar.nested_sampling.utils import get_parameter_masks
from fm4ar.priors.config import PriorConfig


def test__get_parameter_masks() -> None:
    """
    Test `get_parameter_masks()`.
    """

    # Case 1: valid configuration
    prior = Vasist2023Prior(random_seed=42)
    config = PriorConfig(
        dataset="vasist_2023",
        parameters={
            "C/O": "infer",
            "Fe/H": "infer",
            "P_quench": "infer",
            "log_X_Fe": "infer",
            "log_X_MgSiO3": "infer",
            "f_sed": "infer",
            "log_K_zz": "infer",
            "sigma_g": "infer",
            "log_g": "marginalize",
            "R_P": "marginalize",
            "T_0": "marginalize",
            "T_3/T_connect": "marginalize",
            "T_2/T_3": "condition = 0.5",
            "T_1/T_2": "condition = 0.5",
            "alpha": "condition = 0.5",
            "log_delta/alpha": "condition = 0.5",
        },
        random_seed=42,
    )

    (
        infer_mask,
        marginalize_mask,
        condition_mask,
        condition_values,
    ) = get_parameter_masks(prior=prior, config=config)

    assert infer_mask.tolist() == 8 * [True] + 8 * [False]
    assert marginalize_mask.tolist() == 8 * [False] + 4 * [True] + 4 * [False]
    assert condition_mask.tolist() == 12 * [False] + 4 * [True]
    assert all(np.isnan(x) for x in condition_values[:12])
    assert all(x == 0.5 for x in condition_values[12:])

    # Case 2: invalid configuration (missing parameter)
    prior = Vasist2023Prior(random_seed=42)
    config = PriorConfig(dataset="vasist_2023", parameters={"A": "infer"})
    with pytest.raises(KeyError) as key_error:
        get_parameter_masks(prior=prior, config=config)
    assert "Parameter 'C/O' not found in the" in str(key_error.value)

    # Case 3: invalid configuration (unknown action)
    prior = Vasist2023Prior(random_seed=42)
    config = PriorConfig(dataset="vasist_2023", parameters={"C/O": "unknown"})
    with pytest.raises(ValueError) as value_error:
        get_parameter_masks(prior=prior, config=config)
    assert "Unknown action 'unknown' for parameter" in str(value_error.value)
