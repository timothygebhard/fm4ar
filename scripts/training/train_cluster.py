"""
Train a model on an HTCondor cluster.

This script is meant to be called from the login node of the cluster,
using the `--start-submission` flag. This will then (on the login node)
create a submission file and launch a new job on the cluster, which will
run *without* the `--start-submission` flag. This job will then run the
actual training, and it will automatically restart itself from the
latest checkpoint if the job runtime limit is reached.
"""

import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

from fm4ar.training.args import get_cli_arguments
from fm4ar.training.preparation import prepare_new, prepare_resume
from fm4ar.training.stages import train_stages
from fm4ar.utils.config import load_config
from fm4ar.utils.git_utils import document_git_status
from fm4ar.utils.htcondor import (
    CondorSettings,
    check_if_on_login_node,
    condor_submit_bid,
    copy_logfiles,
    create_submission_file,
)


if __name__ == "__main__":

    print("\nTRAIN MODEL ON HTCONDOR CLUSTER\n")

    # Get arguments and load the experiment configuration
    args = get_cli_arguments()
    config = load_config(args.experiment_dir)

    # Make sure we don't try to run the training on the login node
    check_if_on_login_node(start_submission=args.start_submission)

    # Get path to this script and add it to the arguments for the job
    job_arguments = [Path(__file__).resolve().as_posix()]

    # -------------------------------------------------------------------------
    # Either prepare first submission...
    # -------------------------------------------------------------------------

    # This branch does not run any training, but only creates the submit file
    if args.start_submission:
        job_arguments.append(f"--experiment-dir {args.experiment_dir}")
        if args.checkpoint_name != "model__latest.pt":
            job_arguments.append(f"--checkpoint-name {args.checkpoint_name}")

    # -------------------------------------------------------------------------
    # ...or actually run the training
    # -------------------------------------------------------------------------

    else:

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

        # Train model (either to completion or until the time limit is reached)
        with threadpool_limits(limits=1, user_api="blas"):
            complete = train_stages(pm=pm, dataset=dataset)

        # Copy log files and append epoch number
        copy_logfiles(
            log_dir=args.experiment_dir / "logs",
            label=f"epoch-{pm.epoch:03d}",
        )

        # Check if training is complete (in which case we do not resubmit)
        if complete:
            print("Training complete! Job will not be resubmitted.\n")
            sys.exit(0)

        # If training is not complete, we need to resubmit the job
        else:
            job_arguments.append(f"--experiment-dir {args.experiment_dir}")

    # -------------------------------------------------------------------------
    # Create next submission file and submit job
    # -------------------------------------------------------------------------

    # Combine condor arguments with the rest of the condor settings
    condor_settings = CondorSettings(**config["local"]["condor"])
    condor_settings.arguments = job_arguments

    # Create submission file
    print("Creating submission file...", end=" ")
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=args.experiment_dir,
    )
    print("Done!\n")

    condor_submit_bid(bid=condor_settings.bid, file_path=file_path)
