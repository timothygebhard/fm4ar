"""
Unit tests for `fm4ar.datasets.scaling`.
"""

import torch

from fm4ar.datasets.scaling import Standardizer, Normalizer, get_theta_scaler
from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER


def test__get_scaler() -> None:
    """
    Test `fm4ar.datasets.get_scaler()`.
    """

    # Case 1: Standardizer
    config = {
        "data": {
            "name": "vasist-2023",
            "theta_scaler": "standardizer",
        }
    }
    scaler = get_theta_scaler(config=config)
    assert isinstance(scaler, Standardizer)
    assert torch.allclose(
        scaler.mean,
        torch.tensor(
            [
                0.85,
                0.0,
                -1.5,
                -0.65,
                -0.65,
                5.0,
                9.0,
                2.025,
                3.75,
                1.45,
                1300.0,
                0.5,
                0.5,
                0.5,
                1.5,
                0.5,
            ]
        )
    )
    assert torch.allclose(
        scaler.std,
        torch.tensor(
            [
                0.4330127018922193,
                0.8660254037844386,
                2.598076211353316,
                0.9526279441628824,
                0.9526279441628824,
                2.8867513459481287,
                2.309401076758503,
                0.562916512459885,
                1.0103629710818451,
                0.31754264805429416,
                577.3502691896257,
                0.28867513459481287,
                0.28867513459481287,
                0.28867513459481287,
                0.28867513459481287,
                0.28867513459481287,
            ]
        )
    )

    # Case 1: Normalizer
    config = {
        "data": {
            "name": "vasist-2023",
            "theta_scaler": "normalizer",
        }
    }
    scaler = get_theta_scaler(config=config)
    assert isinstance(scaler, Normalizer)
    assert torch.allclose(
        scaler.minimum,
        torch.tensor(LOWER).float(),
    )
    assert torch.allclose(
        scaler.maximum,
        torch.tensor(UPPER).float(),
    )
