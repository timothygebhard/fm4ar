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
from typing import Generator


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
