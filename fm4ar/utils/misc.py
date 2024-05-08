"""
Miscellaneous functions that don't fit anywhere else.
"""

import os
from contextlib import redirect_stdout, contextmanager
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
def suppress_output() -> Generator[None, None, None]:
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield
