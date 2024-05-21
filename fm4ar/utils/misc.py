"""
Miscellaneous functions that don't fit anywhere else.
"""

import os
from contextlib import (
    contextmanager,
    nullcontext,
    redirect_stdout,
    redirect_stderr,
)
from functools import wraps
from typing import Any, Callable, Generator

from frozendict import frozendict


def freeze_args(func: Callable) -> Callable:
    """
    Decorator to freeze the arguments of a function.

    In particular, this can be used to convert a dict argument (which
    is not hashable) into a frozendict (which is hashable) to allow
    caching a function whose arguments include a dict.

    Source: https://stackoverflow.com/a/53394430/4100721
    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        args = tuple(
            [frozendict(arg) if isinstance(arg, dict) else arg for arg in args]
        )
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


@contextmanager
def suppress_output(
    stdout: bool = True,
    stderr: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager to suppress the output to stdout and/or stderr.

    Args:
        stdout: Whether to suppress the output to stdout.
        stderr: Whether to suppress the output to stderr.
    """

    # Note: `with suppress_stdout and suppress_stderr` does *not* work
    with open(os.devnull, "w") as null:
        suppress_stdout = redirect_stdout(null) if stdout else nullcontext()
        suppress_stderr = redirect_stderr(null) if stderr else nullcontext()
        with suppress_stdout:
            with suppress_stderr:
                yield
