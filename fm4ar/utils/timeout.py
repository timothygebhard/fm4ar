"""
Methods for handling timeouts (= limiting the runtime of a function).
"""

from contextlib import contextmanager
from signal import SIGALRM, alarm, signal
from typing import Any, Iterator


class TimeoutException(Exception):
    pass


@contextmanager
def timelimit(seconds: int) -> Iterator[None]:
    """
    Context manager that raises a TimeoutException if the code block
    takes longer than `seconds` to execute.

    Source: https://stackoverflow.com/a/601168/4100721

    Args:
        seconds: The maximum number of seconds to allow the code block.
    """

    def signal_handler(_: Any, __: Any) -> None:
        raise TimeoutException("Timed out!")

    signal(SIGALRM, signal_handler)
    alarm(seconds)

    try:
        yield
    finally:
        alarm(0)
