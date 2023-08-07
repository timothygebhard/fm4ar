"""
Instantiate a model from a config file and count the number of
parameters without starting a training run.
"""

from argparse import ArgumentParser
from time import time
from pathlib import Path

from fm4ar.training.preparation import prepare_new
from fm4ar.utils.config import load_config
from fm4ar.utils.torchutils import get_number_of_model_parameters


if __name__ == "__main__":

    script_start = time()
    print("\nCOUNT PARAMETERS OF MODEL\n")

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory with config.yaml.",
    )
    args = parser.parse_args()

    print(f"Experiment directory:", flush=True)
    print(f"{args.experiment_dir.resolve()}\n", flush=True)

    # Load config and update local settings to ensure they work on macOS.
    # Also, load only the smaller test set to get the correct theta_dim.
    config = load_config(args.experiment_dir)
    config["data"]["which"] = "test"
    config["local"]["wandb"] = False
    config["local"]["device"] = "cpu"
    config["local"]["n_workers"] = 0

    # Load data and build model (needed to infer theta_dim and context_dim)
    pm, dataset = prepare_new(
        experiment_dir=args.experiment_dir,
        config=config,
    )
    print("\n")

    for name, model in (
        ("total", pm.network),
        ("context embedding net", pm.network.context_embedding_net),
    ):
        n_trainable = get_number_of_model_parameters(model, (True,))
        n_fixed = get_number_of_model_parameters(model, (False,))
        n_total = n_trainable + n_fixed
        print(f"Number of {name} parameters:", flush=True)
        print(f"n_trainable: {n_trainable:,}", flush=True)
        print(f"n_fixed:     {n_fixed:,}", flush=True)
        print(f"n_total:     {n_total:,}\n", flush=True)

    print(f"\nThis took {time() - script_start:.2f} seconds!\n", flush=True)
