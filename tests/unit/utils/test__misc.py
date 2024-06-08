"""
Unit tests for `fm4ar.utils.misc`.
"""

import sys

import pytest

from fm4ar.utils.misc import suppress_output


def test__suppress_output(capsys: pytest.CaptureFixture) -> None:
    """
    Test 'fm4ar.utils.misc.suppress_output'.
    """

    # Case 1: Suppress stdout
    with suppress_output(stdout=True, stderr=False):
        print("This should not be printed!")
        print("But this should!", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "But this should!\n"

    # Case 2: Suppress stderr
    with suppress_output(stdout=False, stderr=True):
        print("This should be printed!")
        print("But this should not!", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == "This should be printed!\n"
    assert captured.err == ""
