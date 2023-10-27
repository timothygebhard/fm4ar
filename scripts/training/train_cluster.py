"""
Train a model on an HTCondor cluster.

The script should be called using the `--start-submission` flag on the
login node. This will create a submission file and submit the job to
the cluster. The job will then be retried automatically if the runtime
limit is reached but the training is not complete yet.
"""

import sys
from socket import gethostname
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
    create_submission_file,
)


if __name__ == "__main__":

    print("\nTRAIN MODEL ON HTCONDOR CLUSTER\n")

    # Get arguments and load the experiment configuration
    args = get_cli_arguments()
    config = load_config(args.experiment_dir)

    # Make sure we don't try to run the training on the login node
    check_if_on_login_node(start_submission=args.start_submission)

    # -------------------------------------------------------------------------
    # Either prepare submission file and submit job...
    # -------------------------------------------------------------------------

    # This branch does not run any training, but only creates the submit file
    if args.start_submission:

        # Collect arguments for job (Python script and command line options)
        job_arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
            f"--checkpoint-name {args.checkpoint_name}"
            "--update-config" if args.update_config else "",
        ]

        # Combine condor settings from config file with job arguments and the
        # options that are required to automatically restart the job if the
        # runtime limit is reached but the training is not complete yet
        condor_settings = CondorSettings(**config["local"]["condor"])
        condor_settings.arguments = job_arguments
        condor_settings.retry_on_exit_code = 42
        condor_settings.log_file_name = "log.$$([NumJobStarts])"

        # Create submission file
        print("Creating submission file...", end=" ")
        file_path = create_submission_file(
            condor_settings=condor_settings,
            experiment_dir=args.experiment_dir,
        )
        print("Done!\n")

        # Submit job to HTCondor cluster and exit
        condor_submit_bid(bid=condor_settings.bid, file_path=file_path)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # ...or actually run the training
    # -------------------------------------------------------------------------

    else:

        print("Running on host:", gethostname(), "\n", flush=True)

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
                update_config=args.update_config,
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

        # If the training is complete, we can end the job. Otherwise, we exit
        # with code 42 (see CondorSettings above), which will cause the job to
        # be put on hold and retried automatically.
        if complete:
            print("Training complete! Ending job.\n")
            sys.exit(0)
        else:
            print("Training incomplete! Sending job back to the queue.\n")
            sys.exit(42)
