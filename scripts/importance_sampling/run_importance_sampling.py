"""
Run importance sampling to get posterior samples for a given spectrum.
This script currently only works for the Vasist-2023 dataset.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from p_tqdm import p_map

from fm4ar.datasets.vasist_2023.prior import Prior, SIGMA
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.importance_sampling.proposals import draw_proposal_samples
from fm4ar.importance_sampling.utils import (
    compute_effective_sample_size,
    compute_is_weights,
    construct_context,
    get_target_spectrum,
)
from fm4ar.utils.hdf import load_merged_hdf_files, save_to_hdf
from fm4ar.utils.htcondor import (
    CondorSettings,
    DAGManFile,
    create_submission_file,
    condor_submit_dag,
)
from fm4ar.utils.multiproc import get_number_of_available_cores


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bid",
        type=int,
        default=25,
        help="Bid for the HTCondor job.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="model__best.pt",
        help=(
            "Name of the checkpoint file that contains the trained model. "
            "Will be ignored when running on nested sampling results."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=0,
        help="Job number for parallel processing; must be in [0, n_jobs).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ml", "nested_sampling", "unconditional_flow"],
        default="ml",
        help="Type of model to assume for importance sampling.",
    )
    parser.add_argument(
        "--n-sampling-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel jobs to use for drawing proposal samples. "
            "These are jobs that will require a GPU for ML models."
        ),
    )
    parser.add_argument(
        "--n-simulation-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel jobs to use for simulating spectra. "
            "These jobs only require CPUs (the more, the better)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of samples to draw from proposal distribution.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="Number of CPUs to request for the HTCondor job.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling from proposal distribution.",
    )
    parser.add_argument(
        "--random-seed-offset",
        type=int,
        default=0,
        help="Offset for the random seed.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1000,
        choices=[400, 1000],
        help="Resolution R = ∆λ/λ of the spectra (default 1000).",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help="If True, create a submission file and launch a job.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "draw_proposal_samples",
            "merge_proposal_samples",
            "simulate_spectra",
            "merge_simulation_results",
        ],
        default=None,
        help="Stage of the importance sampling workflow that should be run.",
    )
    parser.add_argument(
        "--target-spectrum",
        type=str,
        default="benchmark",
        help=(
            "Target spectrum. If 'benchmark', use the benchmark spectrum "
            "from Vasist et al. (2023). If a number N is given, use the N-th "
            "spectrum from the evaluation set."
        )
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-4,
        help="Tolerance parameter for FM models; ignored otherwise.",
    )
    args = parser.parse_args()

    return args


def process_theta_i(theta_i: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Returns the (raw) weight, likelihood, prior, and model probability.
    """

    # NOTE: We do not pass the `simulator`, `context`, ... as arguments to
    # this function because it seems like in that case, the multiprocessing
    # parallelization does not work properly (there is only ever one process
    # at a time doing work). This function can therefore only be called when
    # the outer scope provides these variables!

    # Evaluate the prior at theta_i
    # If the prior is 0, we can skip the rest (because the weight will be 0)
    if (prior_value := prior.evaluate(theta_i)) == 0:
        return np.full(n_bins, np.nan), 0.0, 0.0

    # Simulate the spectrum that belongs to theta_i
    # If the simulation fails, we simply set the weight to 0.
    result = simulator(theta_i)
    if result is None:
        return np.full(n_bins, np.nan), 0.0, 0.0
    else:
        _, x_i = result

    # Compute the likelihood
    likelihood = float(
        np.exp(float(-0.5 * np.sum(((x_i - x_0) / SIGMA) ** 2)))
    )

    return x_i, likelihood, prior_value


def prepare_and_launch_dagman_file(
    args: argparse.Namespace,
    output_dir: Path,
) -> None:

    # Initialize a new DAGMan file
    dag = DAGManFile()

    # Collect arguments that will be shared between all stages
    shared_arguments = [
        Path(__file__).resolve().as_posix(),
        f"--experiment-dir {args.experiment_dir}",
        f"--model-type {args.model_type}",
        f"--random-seed {args.random_seed}",
        f"--random-seed-offset {args.random_seed_offset}",
    ]

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------

    # We only need to request a GPU if we are using ML-based model
    num_gpus = 1 if args.model_type in ["ml", "unconditional_flow"] else 0

    # Add extra arguments needed for this stage
    arguments = [
        *shared_arguments,
        f"--n-samples {args.n_samples}",
        f"--resolution {args.resolution}",
        "--stage draw_proposal_samples",
    ]

    # Create submission file
    condor_settings = CondorSettings(
        num_cpus=args.num_cpus,
        memory_cpus=args.num_cpus * 1000,
        num_gpus=num_gpus,
        memory_gpus=15_000,
        arguments=arguments,
        log_file_name="draw_proposal_samples.$(Process)",
        queue=args.n_jobs,
    )
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=output_dir,
        file_name="draw_proposal_samples.sub",
    )

    # Add the job to the DAGMan file
    dag.add_job(
        name="draw_proposal_samples",
        file_path=file_path,
    )

    # -------------------------------------------------------------------------
    # Stage 2: Merge the samples from the proposal distribution
    # -------------------------------------------------------------------------

    # Add extra arguments needed for this stage
    arguments = [
        *shared_arguments,
        "--stage merge_proposal_samples",
    ]

    # Create submission file
    condor_settings = CondorSettings(
        num_cpus=2,
        memory_cpus=args.num_cpus * 1000,
        arguments=arguments,
        log_file_name="merge_proposal_samples.$(Process)",
    )
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=output_dir,
        file_name="merge_proposal_samples.sub",
    )

    # Add the job to the DAGMan file
    dag.add_job(
        name="merge_proposal_samples",
        file_path=file_path,
        depends_on=["draw_proposal_samples"],
    )

    # -------------------------------------------------------------------------
    # Stage 3: Simulate spectra for the proposal samples
    # -------------------------------------------------------------------------

    # Add extra arguments needed for this stage
    arguments = [
        *shared_arguments,
        f"--resolution {args.resolution}",
        "--stage simulate_spectra",
    ]

    # Create submission file
    condor_settings = CondorSettings(
        num_cpus=args.num_cpus,
        memory_cpus=args.num_cpus * 1000,
        arguments=arguments,
        log_file_name="simulate_spectra.$(Process)",
    )
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=output_dir,
        file_name="simulate_spectra.sub",
    )

    # Add the job to the DAGMan file
    dag.add_job(
        name="simulate_spectra",
        file_path=file_path,
        depends_on=["merge_proposal_samples"],
    )

    # -------------------------------------------------------------------------
    # Stage 4: Merge the results from all jobs and compute the weights
    # -------------------------------------------------------------------------

    # Add extra arguments needed for this stage
    arguments = [
        *shared_arguments,
        "--stage merge_simulation_results",
    ]

    # Create submission file
    condor_settings = CondorSettings(
        num_cpus=2,
        memory_cpus=args.num_cpus * 1000,
        arguments=arguments,
        log_file_name="merge_simulation_results.$(Process)",
    )
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=output_dir,
        file_name="merge_simulation_results.sub",
    )

    # Add the job to the DAGMan file
    dag.add_job(
        name="simulate_spectra",
        file_path=file_path,
        depends_on=["simulate_spectra"],
    )

    # -------------------------------------------------------------------------
    # Submit the DAGMan file to HTCondor
    # -------------------------------------------------------------------------

    # Save the DAGMan file
    file_path = output_dir / "importance_sampling.dag"
    dag.save(file_path=file_path)

    # Submit the DAGMan file to HTCondor
    condor_submit_dag(file_path=file_path, verbose=True)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN IMPORTANCE SAMPLING\n")

    # Get the command line arguments
    args = get_cli_arguments()

    # Define and prepare ouput directory
    dir_name = (
        "benchmark" if args.target_spectrum == "benchmark"
        else f"{int(args.target_spectrum):04d}"
    )
    output_dir = args.experiment_dir / "importance_sampling" / dir_name
    output_dir.mkdir(exists_ok=True)

    # Set the random seed (both for numpy and torch)
    effective_random_seed = args.random_seed + args.random_seed_offset
    np.random.seed(effective_random_seed)
    torch.manual_seed(effective_random_seed)

    # -------------------------------------------------------------------------
    # If --start-submission: Create DAG file, launch job, and exit
    # -------------------------------------------------------------------------

    if args.start_submission:
        prepare_and_launch_dagman_file(args=args, output_dir=output_dir)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Otherwise, prepare to run one (or multiple) workflow stages
    # -------------------------------------------------------------------------

    if args.stage == "draw_proposal_samples" or args.stage is None:

        print("Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Get the wavelengths and the flux of the target spectrum
        simulator = Simulator(noisy=False, R=args.resolution)
        wlen, x_0 = get_target_spectrum(
            args=args,
            output_dir=output_dir,
            simulator=simulator,
        )

        # Construct context (only relevant for ML-based models)
        context = construct_context(x_0=x_0, wlen=wlen, SIGMA=SIGMA)

        # Draw samples from proposal distribution and save them
        theta, probs = draw_proposal_samples(args=args, context=context)
        save_to_hdf(
            file_path=output_dir / f"proposal-samples-{args.job:04d}.hdf",
            theta=theta,
            probs=probs,
        )

    elif args.stage == "merge_proposal_samples" or args.stage is None:

        print("Merge samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Merge the results from all jobs and save them into a single HDF file
        merged = load_merged_hdf_files(
            target_dir=output_dir,
            name_pattern="proposal-samples-*.hdf",
            keys=["theta", "probs"],
        )
        save_to_hdf(
            file_path=output_dir / "proposal-samples.hdf",
            theta=merged["theta"].astype(np.float32),
            probs=merged["probs"].astype(np.float32),
        )

        # TOOD: Delete the individual files

    elif args.stage == "simulate_spectra" or args.stage is None:

        print("Simulate spectra for theta_i", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Load the theta samples and probabilities
        proposal_samples = load_merged_hdf_files(
            target_dir=output_dir,
            name_pattern="proposal-samples.hdf",
            keys=["theta", "probs"],
        )

        # Select the samples that belong to the current job
        theta = proposal_samples["theta"][args.job::args.n_jobs]
        probs = proposal_samples["probs"][args.job::args.n_jobs]

        # Set up prior and simulator
        prior = Prior(random_seed=effective_random_seed)
        simulator = Simulator(noisy=False, R=args.resolution)

        # Get the wavelengths and the flux of the target spectrum
        wlen, x_0 = get_target_spectrum(args=args, output_dir=output_dir)
        n_bins = len(wlen)

        # Compute spectra, likelihoods and prior values (in parallel)
        print("Simulating spectra (in parallel):", flush=True)
        num_cpus = get_number_of_available_cores()
        results = p_map(process_theta_i, theta, num_cpus=num_cpus, ncols=80)
        print()

        # Unpack the results from the parallel map
        x, likelihoods, prior_values = zip(*results, strict=True)
        x = np.array(x)
        likelihoods = np.array(likelihoods).flatten()
        prior_values = np.array(prior_values).flatten()

        # Drop everything that has a prior of 0 (i.e., is outside the bounds),
        # or where the simulation failed (i.e., the spectrum contains NaNs)
        mask = np.logical_and(prior_values > 0, ~np.isnan(x).any(axis=1))
        theta = theta[mask]
        probs = probs[mask]
        x = x[mask]
        likelihoods = likelihoods[mask]
        prior_values = prior_values[mask]
        n = len(theta)
        print(f"Dropped {np.sum(~mask):,} invalid samples!")
        print(f"Remaining samples: {n:,} ({100 * n / args.n_samples:.2f}%)\n")

        # Save the results for the current job
        file_name = f"simulations-{args.job:04d}.hdf"
        save_to_hdf(
            file_path=output_dir / file_name,
            theta=theta.astype(np.float32),
            probs=probs.astype(np.float64),
            x=x.astype(np.float32),
            likelihoods=likelihoods.astype(np.float64),
            prior_values=prior_values.astype(np.float32),
        )

    elif args.stage == "merge_simulation_results" or args.stage is None:

        print("Merge simulation results and compute weights", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Merge the results from all jobs and load them
        merged = load_merged_hdf_files(
            target_dir=output_dir,
            name_pattern="simulations-*.hdf",
            keys=["theta", "probs", "x", "likelihoods", "prior_values"],
        )

        # Compute the importance sampling weights
        raw_weights, weights = compute_is_weights(
            likelihoods=merged["likelihoods"],
            prior_values=merged["prior_values"],
            probs=merged["probs"],
        )
        merged["raw_weights"] = raw_weights.astype(np.float64)
        merged["weights"] = weights.astype(np.float64)

        # Compute the effective sample size and sample efficiency
        n_eff, sample_efficiency = compute_effective_sample_size(weights)
        print(f"Effective sample size: {n_eff:.2f}")
        print(f"Sample efficiency:     {100 * sample_efficiency:.2f}%\n")

        # Load the target spectrum and add it to the merged results
        wlen, x_0 = get_target_spectrum(args=args, output_dir=output_dir)
        merged["wlen"] = wlen.astype(np.float32)
        merged["x_0"] = x_0.astype(np.float32)

        # Save the final results
        print("Saving results...", end=" ")
        save_to_hdf(
            file_path=output_dir / "importance_sampling_results.hdf",
            **merged,
        )
        print("Done!")

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
