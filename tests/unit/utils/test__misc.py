"""
Unit tests for `fm4ar.utils.misc`.
"""

import sys
from functools import lru_cache
from typing import Any

import pytest
from frozendict import frozendict

from fm4ar.utils.misc import freeze_args, suppress_output


def test__freeze_args() -> None:
    """
    Test 'fm4ar.utils.misc.freeze_args'.
    """

    # Case 1: We can use `freeze_args` as a decorator to freeze the arguments
    @freeze_args
    def dummy_function_1(x: dict) -> Any:
        return x

    assert dummy_function_1({"a": 1}) == frozendict({"a": 1})

    # Case 2: We can use `freeze_args` in conjunction with `lru_cache`
    @freeze_args
    @lru_cache
    def dummy_function_2(x: dict) -> Any:
        return x

    assert dummy_function_2({"a": 1}) == frozendict({"a": 1})

    # Case 3: Check that just `lru_cache` without `freeze_args` does NOT work
    @lru_cache
    def dummy_function_3(x: dict) -> Any:
        return x

    with pytest.raises(TypeError) as type_error:
        dummy_function_3({"a": 1})  # type: ignore
    assert "unhashable type: 'dict'" in str(type_error.value)


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
