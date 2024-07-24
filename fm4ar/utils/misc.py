"""
Miscellaneous functions that don't fit anywhere else.
"""

import os
import platform
import re
import subprocess
from contextlib import (
    contextmanager,
    nullcontext,
    redirect_stderr,
    redirect_stdout,
)
from typing import Generator


def get_processor_name() -> str:
    """
    Auxiliary function to get the name of the processor.
    Source: https://stackoverflow.com/a/13078519/4100721
    """

    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return str(subprocess.check_output(cmd).strip().decode())

    elif platform.system() == "Linux":
        cmd = ["cat", "/proc/cpuinfo"]
        all_info = subprocess.check_output(cmd).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(
                    pattern=r".*model name.*:",
                    repl="",
                    string=line,
                    count=1,
                )

    raise NotImplementedError("Unsupported platform!")


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
        with suppress_stdout:  # noqa: SIM117
            with suppress_stderr:
                yield
