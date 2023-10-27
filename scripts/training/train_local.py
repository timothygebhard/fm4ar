"""
Train a model locally.
"""

import sys

from threadpoolctl import threadpool_limits

from fm4ar.training.args import get_cli_arguments
from fm4ar.training.preparation import prepare_new, prepare_resume
from fm4ar.training.stages import train_stages
from fm4ar.utils.config import load_config
from fm4ar.utils.git_utils import document_git_status


if __name__ == "__main__":

    print("\nTRAIN MODEL ON LOCAL MACHINE\n")

    # Get arguments and load the experiment configuration
    args = get_cli_arguments()
    config = load_config(args.experiment_dir)

    # Document the status of the git repository
    document_git_status(target_dir=args.experiment_dir, verbose=True)

    # Check if there exists a checkpoint file from which we can resume
    checkpoint_file_path = args.experiment_dir / args.checkpoint_name
    if checkpoint_file_path.exists():
        print("Checkpoint found, resuming training run!")
        pm, dataset = prepare_resume(
            experiment_dir=args.experiment_dir,
            checkpoint_name=args.checkpoint_name,
            config=config,
        )
        print()

    # If no checkpoint file exists, we need to start from scratch
    else:
        print("No checkpoint found, starting new training run!")
        pm, dataset = prepare_new(
            experiment_dir=args.experiment_dir,
            config=config,
        )
        print()

    # Train model (either to completion, or until a time limit is reached,
    # or until the early stopping criterion is met)
    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm=pm, dataset=dataset)

    # Print a message to indicate whether training was completed or not
    if complete:
        print("All training stages complete!\n")
    else:
        print("Program terminated due to runtime limit!\n")

    sys.exit(0)
