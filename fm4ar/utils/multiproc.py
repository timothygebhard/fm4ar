"""
Utility functions for multiprocessing.
"""

import os


def get_number_of_available_cores(default: int = 8) -> int:
    """
    Get the number cores available to the current process (if possible,
    otherwise return the given default value).

    Args:
        default: The default number of cores that is returned if
            ``os.sched_getaffinity()`` is not available.

    Returns:
        The number of cores available to the current process.
    """

    try:
        return len(os.sched_getaffinity(0))  # type: ignore
    except AttributeError:
        return default
