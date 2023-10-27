"""
Argument parser for training scripts.
"""

import argparse
from pathlib import Path


def get_cli_arguments() -> argparse.Namespace:  # pragma: no cover
    """
    Argument parser for training scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model__latest.pt",
        help="Name of checkpoint file from which to resume training."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Directory containing the experiment configuration.",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help=(
            "If this flag is used, the script will prepare the HTCondor "
            "submission file and launch a new job (but not actually run the "
            "training itself). Ignored for local training."
        ),
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help=(
            "If this flag is used, the configuration in the model checkpoint "
            "will be replaced with the configuration from the experiment dir. "
            "This can be useful, e.g., if there was a bug in the config, or "
            "if you want to increase the number of epochs, or ..."
        ),
    )
    args = parser.parse_args()

    return args
