"""
Create a HTCondor submit file and launch a new nested sampling run.
"""

import sys
from time import time
from pathlib import Path

from coolname import generate_slug

from fm4ar.nested_sampling import get_argument_parser
from fm4ar.utils.htcondor import create_submission_file, condor_submit_bid


if __name__ == "__main__":

    script_start = time()
    print("\nLAUNCH NESTED SAMPLING RUN ON HTCONDOR CLUSTER \n", flush=True)

    # Get command line arguments
    parser = get_argument_parser()
    parser.add_argument(
        "--bid",
        type=int,
        default=25,
        help="Bid to use for the HTCondor job.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=98_304,
        help="Memory (in MB) to use for the HTCondor job.",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=96,
        help="Number of CPUs to use for the HTCondor job.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["dynesty", "nautilus"],
        default="nautilus",
        help="Nested sampling sampler to use.",
    )
    args = parser.parse_args()

    # Get a list of arguments to ignore when creating the HTCondor job
    ignored = ["bid", "memory", "n_cpus", "sampler"]

    # Create a new directory for this run
    print("Creating new run directory...", end=" ", flush=True)
    results_dir = Path(__file__).parent / args.sampler / "results"
    results_dir.mkdir(exist_ok=True)
    n_runs = len(list(results_dir.glob("*")))
    run_dir = results_dir / str(f"{n_runs:d}-" + generate_slug(2))
    run_dir.mkdir(exist_ok=True)
    print("Done!", flush=True)

    print(f"\nRun directory: {run_dir}\n", flush=True)

    # Collect arguments
    print("Collecting arguments for job...", end=" ", flush=True)
    script_path = (
        Path(__file__).parent
        / args.sampler
        / f"run_{args.sampler}.py"
    ).as_posix()
    arguments = f"{script_path} --run-dir {run_dir.as_posix()} "
    for key, value in args.__dict__.items():
        if key == "add_noise" and value:
            arguments += "--add-noise "
        elif key not in ignored and value is not None:
            if isinstance(value, list):
                value = " ".join([f"'{v}'" for v in value])
            arguments += f"--{key.replace('_', '-')} {value} "
    print("Done!", flush=True)

    # Create a HTCondor submit file
    print("Creating submission file...", end=" ", flush=True)
    file_path = create_submission_file(
        condor_settings={
            "executable": sys.executable,
            "num_cpus": args.n_cpus,
            "num_gpus": 0,
            "memory_cpus": args.memory,
            "memory_gpus": 0,
            "arguments": arguments,
        },
        experiment_dir=run_dir,
    )
    print("Done!", flush=True)

    # Submit the job
    print("Submitting job...", end=" ", flush=True)
    condor_submit_bid(file_path=file_path, bid=args.bid)
    print("Done!", flush=True)

    print(f"\nThis took {time() - script_start:.2f} seconds.\n", flush=True)
