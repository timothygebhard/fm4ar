"""
Tests for `fm4ar.utils.tracking`.
"""

from time import sleep

import numpy as np

from fm4ar.utils.tracking import (
    AvgTracker,
    RuntimeLimits,
)


def test__avg_tracker() -> None:
    """
    Test `fm4ar.utils.tracking.AvgTracker`.
    """

    avg_tracker = AvgTracker()

    # Case 1: empty tracker
    assert np.isnan(avg_tracker.get_avg())
    assert np.isnan(avg_tracker.get_last())

    # Case 2: add some values
    avg_tracker.update(1.0)
    avg_tracker.update(2.0)
    avg_tracker.update(3.0)
    assert avg_tracker.get_avg() == 2.0
    assert avg_tracker.get_last() == 3.0

    # Case 3: add some more values
    avg_tracker.update(4.0)
    avg_tracker.update(5.0)
    avg_tracker.update(6.0)
    assert avg_tracker.get_avg() == 3.5


def test__runtime_limits() -> None:
    """
    Test `fm4ar.utils.tracking.RuntimeLimits`.
    """

    # Case 1: Check limits on epochs
    runtime_limits = RuntimeLimits(max_epochs=10)
    assert runtime_limits.max_epochs == 10
    assert runtime_limits.max_runtime is None
    assert runtime_limits.max_epochs_exceeded(100)
    assert runtime_limits.limits_exceeded(100)
    assert not runtime_limits.limits_exceeded(0)
    assert not runtime_limits.max_runtime_exceeded()

    # Case 2: Check limits on runtime
    runtime_limits = RuntimeLimits(max_runtime=1.0)
    assert runtime_limits.max_epochs is None
    sleep(1.5)
    assert runtime_limits.max_runtime_exceeded()
    assert runtime_limits.limits_exceeded(0)
    assert runtime_limits.limits_exceeded(100)
    assert not runtime_limits.max_epochs_exceeded(0)
    assert not runtime_limits.max_epochs_exceeded(100)
