"""
Tests for `fm4ar.utils.multiproc`.
"""

import os

from pytest import MonkeyPatch

from fm4ar.utils.multiproc import get_number_of_available_cores


def test__get_number_of_available_cores(monkeypatch: MonkeyPatch) -> None:
    """
    Test `fm4ar.utils.multiproc.get_number_of_available_cores()`.
    """

    # Case 1: Assume os.sched_getaffinity() is available
    # We need to monkeypatch this to make sure the test works on macOS
    with monkeypatch.context() as context:
        context.setattr(
            target=os,
            name="sched_getaffinity",
            value=lambda _: 13 * [0],
            raising=False,
        )
        assert get_number_of_available_cores() == 13

    # Case 2: Assume os.sched_getaffinity() is not available
    with monkeypatch.context() as context:
        context.delattr(
            target=os,
            name="sched_getaffinity",
            raising=False,
        )
        assert get_number_of_available_cores(default=17) == 17
