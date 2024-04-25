"""
Unit tests for `fm4ar.torchutils.early_stopping`.
"""

from fm4ar.torchutils.early_stopping import (
    EarlyStoppingConfig,
    early_stopping_criterion_reached,
)


def test__early_stopping_criterion_reached() -> None:
    """
    Unit test for `early_stopping_criterion_reached()`.
    """

    # Case 1: no early stopping
    early_stopping_config = EarlyStoppingConfig(enabled=False)
    assert not early_stopping_criterion_reached(
        loss_history=[1.0, 0.9, 0.8, 0.7],
        stage_epoch=4,
        early_stopping_config=early_stopping_config,
    )

    # Case 2: stage-based early stopping (no offset)
    early_stopping_config = EarlyStoppingConfig(stage_patience=2)
    assert not early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7],
        stage_epoch=4,
        early_stopping_config=early_stopping_config,
    )
    assert early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 1.0, 0.6, 0.8, 0.7],
        stage_epoch=4,
        early_stopping_config=early_stopping_config,
    )

    # Case 3: stage-based early stopping (with offset)
    early_stopping_config = EarlyStoppingConfig(
        stage_patience=2,
        stage_offset=2,
    )
    assert not early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 1.0, 0.1, 1.0, 0.9, 0.8, 0.7],
        stage_epoch=6,
        early_stopping_config=early_stopping_config,
    )
    assert early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 1.0, 0.1, 1.0, 0.5, 0.8, 0.7],
        stage_epoch=6,
        early_stopping_config=early_stopping_config,
    )

    # Case 4: global early stopping (no offset)
    early_stopping_config = EarlyStoppingConfig(global_patience=2)
    assert not early_stopping_criterion_reached(
        loss_history=[1.2, 1.1, 1.0, 0.9, 0.8, 0.7],
        stage_epoch=15,
        early_stopping_config=early_stopping_config,
    )
    assert early_stopping_criterion_reached(
        loss_history=[1.2, 1.1, 1.0, 0.5, 0.8, 0.7],
        stage_epoch=15,
        early_stopping_config=early_stopping_config,
    )

    # Case 5: global early stopping (with offset)
    early_stopping_config = EarlyStoppingConfig(
        global_patience=2,
        global_offset=4,
    )
    assert not early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.7],
        stage_epoch=15,
        early_stopping_config=early_stopping_config,
    )
    assert early_stopping_criterion_reached(
        loss_history=[0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.7],
        stage_epoch=15,
        early_stopping_config=early_stopping_config,
    )
