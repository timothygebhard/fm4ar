from collections import OrderedDict
from pathlib import Path

import torch
from pydantic import BaseModel


def get_weights_from_pt_file(
    file_path: Path,
    state_dict_key: str,
    prefix: str,
    drop_prefix: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """
    Load the weights that starts with `prefix` from a *.pt file.

    Args:
        file_path: Path to the *.pt file.
        state_dict_key: Key of the state dict in the *.pt file that
            contains the weights. Usually, this is "model_state_dict".
        prefix: Prefix that the weights must start with. Usually, this
            is the name of a model component, e.g., `vectorfield_net`.
        drop_prefix: Whether to drop the prefix from the keys of the
            returned dictionary.

    Returns:
        An OrderecDict with the weights that can be loaded into a model.
    """

    # Load the full checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(file_path, map_location=device)

    # Select the state dict that contains the weights
    state_dict = checkpoint[state_dict_key]

    # Get the weights that start with `prefix`
    weights = OrderedDict(
        (key if not drop_prefix else key.removeprefix(prefix + "."), value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )

    return weights


def load_and_or_freeze_model_weights(
    model: torch.nn.Module,
    freeze_weights: bool = False,
    load_weights: dict | None = None,
) -> None:
    """
    Load and / or freeze weights of the given model, if requested.

    Args:
        model: The model to be modified.
        freeze_weights: Whether to freeze all weights of the model.
        load_weights: A dictionary with the following keys:
            - `file_path`: Path to the checkpoint file (`*.pt`).
            - `state_dict_key`: Key of the state dict in the checkpoint
                file that contains the weights. Usually, this is
                "model_state_dict".
            - `prefix`: Prefix that the weights must start with.
                Usually, this is the name of a model component, e.g.,
                "vectorfield_net" or "context_embedding_net".
            - `drop_prefix`: Whether to drop the prefix from the keys.
                Default is `True`.
            If `None` or `{}` is passed, no weights are loaded.
    """

    # Load weights, if requested
    if load_weights is not None and load_weights:

        # Validator for the `load_weights` dictionary
        # Seems cleaner than a lot of `if` statements and ValueErrors?
        class LoadWeightsConfig(BaseModel):
            file_path: Path
            state_dict_key: str
            prefix: str
            drop_prefix: bool = True

        # Validate the `load_weights` dictionary
        load_weights_config = LoadWeightsConfig(**load_weights)

        # Load model weights from a file, if requested
        state_dict = get_weights_from_pt_file(
            file_path=load_weights_config.file_path,
            state_dict_key=load_weights_config.state_dict_key,
            prefix=load_weights_config.prefix,
            drop_prefix=load_weights_config.drop_prefix,
        )
        model.load_state_dict(state_dict)

    # Freeze weights, if requested
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
