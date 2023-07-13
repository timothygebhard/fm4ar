import os
from os.path import join

import numpy as np
import yaml
import argparse
import textwrap


from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import autocomplete_model_kwargs

from dingo.core.posterior_models.base_model import Base
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from atmospheric_retrieval.dataset import generate_dataset


def prepare_training_new(train_settings: dict, train_dir: str, local_settings: dict):
    """
    Based on a settings dictionary, initialize a PosteriorModel.

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)

    Returns
    -------
    (WaveformDataset, Base)
    """

    # This modifies the model settings in-place.
    # TODO: set input and context dim here
    autocomplete_model_kwargs(train_settings["model"], input_dim=0, context_dim=0)

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(train_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings={"train_settings": train_settings},
        device=local_settings["device"],
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                config=train_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    # TODO: create AR dataset here
    dataset = generate_dataset(train_settings["data"], local_settings["device"])
    return pm, dataset


def prepare_training_resume(checkpoint_name, local_settings, train_dir):
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    (Base, WaveformDataset)
    """

    pm = build_model_from_kwargs(
        filename=checkpoint_name, device=local_settings["device"]
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="must",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    # TODO: create dataset here
    dataset = generate_dataset(pm.metadata["train_settings"]["data"], local_settings["device"])
    return pm, dataset


def initialize_stage(pm, dataset, stage, num_workers, resume=False):

    train_settings = pm.metadata["train_settings"]

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        dataset,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    n_grad = get_number_of_model_parameters(pm.network, (True,))
    n_nograd = get_number_of_model_parameters(pm.network, (False,))
    print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader


def train_stages(pm, dataset, train_dir, local_settings):
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : Base
    dataset : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    local_settings : dict

    Returns
    -------
    bool
        True if all stages are complete
        False otherwise
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
    )

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stages]))

    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            print(f"\nBeginning training stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, dataset, stage, local_settings["num_workers"], resume=False
            )
        else:
            print(f"\nResuming training in stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, dataset, stage, local_settings["num_workers"], resume=True
            )

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        if pm.epoch == end_epochs[n]:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)
            evaluate = train_settings["training"].get("evaluate", False)
            evaluation_data_path = train_settings["training"].get(
                "evaluation_data_path", None
            )
            if evaluate:
                print("Evaluating model")
                data_dir = join(train_dir, "evaluation")
                # TODO: put evaluation routine here

        if runtime_limits.local_limits_exceeded(pm.epoch):
            print("Local runtime limits reached. Ending program.")
            break

    if pm.epoch == end_epochs[-1]:
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for gravitational-wave single-event inference.
        
        This program can be called in one of two ways:
            a) with a settings file. This will create a new network based on the 
            contents of the settings file.
            b) with a checkpoint file. This will resume training from the checkpoint.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        help="YAML file containing training settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        pm, dataset = prepare_training_new(train_settings, args.train_dir, local_settings)

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, dataset = prepare_training_resume(
            args.checkpoint, local_settings, args.train_dir
        )

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, dataset, args.train_dir, local_settings)

    if complete:
        if args.exit_command:
            print(
                f"All training stages complete. Executing exit command: {args.exit_command}."
            )
            os.system(args.exit_command)
        else:
            print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")


if __name__ == "__main__":
    train_local()
