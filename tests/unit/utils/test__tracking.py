"""
Tests for `fm4ar.utils.tracking`.
"""

import numpy as np

from fm4ar.utils.tracking import (
    AvgTracker,
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
