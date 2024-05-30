"""
Run importance sampling using a trained ML model: either a "proper"
posterior  model (FMPE or NPE), or an unconditional flow model (e.g.,
for creating reference posteriors).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from p_tqdm import p_map

from fm4ar.importance_sampling.config import (
    ImportanceSamplingConfig,
    load_config,
)
from fm4ar.importance_sampling.proposals import draw_proposal_samples
from fm4ar.importance_sampling.target_spectrum import load_target_spectrum
from fm4ar.importance_sampling.utils import (
    compute_effective_sample_size,
    compute_is_weights,
    compute_log_evidence,
)
from fm4ar.likelihoods import get_likelihood_distribution
from fm4ar.priors import get_prior
from fm4ar.simulators import get_simulator
from fm4ar.utils.hdf import load_from_hdf, merge_hdf_files, save_to_hdf
from fm4ar.utils.htcondor import (
    DAGManFile,
    HTCondorConfig,
    condor_submit_dag,
    check_if_on_login_node,
    create_submission_file,
)
from fm4ar.utils.multiproc import get_number_of_available_cores


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the directory containing the trained model.",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=0,
        help="Job number for parallel processing; must be in [0, n_jobs).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs. Default: 1.",
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
        "--start-submission",
        action="store_true",
        help="If True, create a submission file and launch a job.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        required=True,
        help=(
            "Path to the directory containing the importance sampling config "
            "file. The importance sampling config file is expected to be "
            "named `importance_sampling.yaml` (to reduce confusion)."
        ),
    )
    args = parser.parse_args()

    return args


def prepare_and_launch_dag(
    args: argparse.Namespace,
    config: ImportanceSamplingConfig,
    working_dir: Path,
) -> None:
    """
    Prepare and launch the DAGMan file for running the importance
    sampling workflow on HTCondor.

    Args:
        args: The command line arguments.
        config: The importance sampling configuration.
        working_dir: The working directory for the IS run.
    """

    # Initialize a new DAGMan file
    dag = DAGManFile()

    # Add jobs for the different stages of the importance sampling workflow
    for i, (stage, depends_on) in enumerate(
        [
            ("draw_proposal_samples", None),
            ("merge_proposal_samples", ["draw_proposal_samples"]),
            ("simulate_spectra", ["merge_proposal_samples"]),
            ("merge_simulation_results", ["simulate_spectra"]),
        ],
        start=1,
    ):

        # Collect HTCondorSettings for the stage
        htcondor_config: HTCondorConfig = getattr(config, stage).htcondor
        htcondor_config.arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
            f"--working-dir {args.working_dir}",
            f"--stage {stage}",
        ]

        # For the stages that require parallel processing, add the job number
        # and the total number of parallel jobs as arguments; if we just take
        # this from the config file, things break down in non-parallel mode
        if stage in ("draw_proposal_samples", "simulate_spectra"):
            htcondor_config.arguments += [
                "--job $(Process)",
                f"--n-jobs {htcondor_config.queue}",
            ]

        # Create submission file
        file_path = create_submission_file(
            htcondor_config=htcondor_config,
            experiment_dir=working_dir,
            file_name=f"{i}__{stage}.sub"
        )

        # Add the job to the DAGMan file
        dag.add_job(
            name=stage,
            file_path=file_path,
            bid=htcondor_config.bid,
            depends_on=depends_on,
        )

    # Save the DAGMan file
    file_path = working_dir / "0__importance_sampling.dag"
    dag.save(file_path=file_path)

    # Submit the DAGMan file to HTCondor
    condor_submit_dag(file_path=file_path, verbose=True)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN IMPORTANCE SAMPLING\n\n")

    # Get the command line arguments and define shortcuts
    args = get_cli_arguments()
    working_dir = args.working_dir

    # Ensure that we do not run compute-heavy jobs on the login node
    check_if_on_login_node(start_submission=args.start_submission)

    # Load the importance sampling config
    config = load_config(experiment_dir=args.working_dir)

    # -------------------------------------------------------------------------
    # If --start-submission: Create DAG file, launch job, and exit
    # -------------------------------------------------------------------------

    if args.start_submission:
        prepare_and_launch_dag(
            args=args,
            config=config,
            working_dir=working_dir,
        )
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "draw_proposal_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Draw samples (this comes with its own progress bar)
        results = draw_proposal_samples(args=args, config=config)

        print("\nSaving results to HDF...", end=" ", flush=True)
        save_to_hdf(
            file_path=working_dir / f"proposal-samples-{args.job:04d}.hdf",
            samples=results["samples"].astype(np.float32),
            log_prob_samples=results["log_prob_samples"].astype(np.float32),
            log_prob_theta_true=results["log_prob_theta_true"],
        )
        print("Done!\n\n")

    # -------------------------------------------------------------------------
    # Stage 2: Merge samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "merge_proposal_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(2) Merge samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        print("Merging HDF files:", flush=True)
        delete_after_merge = config.merge_proposal_samples.delete_after_merge
        merge_hdf_files(
            target_dir=working_dir,
            name_pattern="proposal-samples-*.hdf",
            output_file_path=working_dir / "proposal-samples.hdf",
            keys=["samples", "log_prob_samples"],
            singleton_keys=["log_prob_theta_true"],
            delete_after_merge=delete_after_merge,
            show_progressbar=True,
        )
        print("\n")

    # -------------------------------------------------------------------------
    # Stage 3: Simulate spectra corresponding to the proposal samples
    # -------------------------------------------------------------------------

    if args.stage == "simulate_spectra" or args.stage is None:

        print(80 * "-", flush=True)
        print("(3) Simulate spectra for theta_i", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Construct the slice of indices for the current job: The current
        # job will process every `n_jobs`-th sample from the proposal samples,
        # starting at an index of `job`. This is useful for parallelization.
        idx = slice(args.job, None, args.n_jobs)

        # Print some information about the number of samples to process
        n_total = config.draw_proposal_samples.n_samples
        n_for_job = len(np.arange(args.job, n_total, args.n_jobs))
        print(f"Total number of samples to process:             {n_total:,}")
        print(f"Number of samples to process for current job:   {n_for_job:,}")
        print()

        # Load the theta samples and probabilities and unpack them
        # We also load the log-probability of the ground truth theta value
        proposal_samples = load_from_hdf(
            file_path=working_dir / "proposal-samples.hdf",
            keys=["samples", "log_prob_samples", "log_prob_theta_true"],
            idx=idx,
        )
        samples = proposal_samples["samples"]
        log_prob_samples = proposal_samples["log_prob_samples"]

        # Sanity check: Are there any duplicate samples? (This can happen if
        # there are issues with setting the random seed for each proposal job)
        n_duplicates = len(np.unique(samples, axis=0)) - len(samples)
        if n_duplicates > 0:
            raise ValueError(
                f"Found {n_duplicates:,} duplicate samples in the proposal!"
            )

        # Load the target spectrum
        target = load_target_spectrum(
            file_path=config.target_spectrum.file_path,
            index=config.target_spectrum.index,
        )
        n_bins = len(target["flux"])

        # Set up prior, simulator, and likelihood distribution
        prior = get_prior(config=config.prior)
        simulator = get_simulator(config=config.simulator)
        likelihood_distribution = get_likelihood_distribution(
            flux_obs=target["flux"],
            config=config.likelihood,
        )

        # Define a function that processes a single `theta_i`
        def process_theta_i(
            theta_i: np.ndarray,
        ) -> tuple[np.ndarray, float, float]:
            """
            Returns the flux, the log-likelihood, and the log-prior
            values for the given `theta_i`.
            """

            # Evaluate the prior at theta_i
            # If the prior is 0, we can skip the rest of the computation since
            # the weight will be 0 anyway. Otherwise, we compute the log-prior
            # value. We return -np.inf here, because we do not want to discard
            # the zero-weight sample, as this would bias the estimate of the
            # log-evidence.
            if (prior_value := prior.evaluate(theta_i)) <= 0:
                return np.full(n_bins, np.nan), -np.inf, -np.inf
            log_prior_value = np.log(prior_value)

            # Simulate the spectrum that belongs to theta_i
            # If the simulation fails, we return NaNs so that we know that
            # we need to discard this sample
            result = simulator(theta_i)
            if result is None:
                return np.full(n_bins, np.nan), np.nan, np.nan
            else:
                _, flux = result

            # Compute the log-likelihood
            # We use the log-likelihood to avoid numerical issues, because
            # the likelihood can take on values on the order of 10^-1000
            log_likelihood = likelihood_distribution.logpdf(flux)

            return flux, log_likelihood, log_prior_value

        # Compute spectra, likelihoods and prior values in parallel
        print("Simulating spectra (in parallel):", flush=True)
        num_cpus = get_number_of_available_cores()
        results = p_map(process_theta_i, samples, num_cpus=num_cpus, ncols=80)
        print()

        # Unpack the results from the parallel map and convert to arrays
        _flux, _log_likelihoods, _log_prior_values = zip(*results, strict=True)
        flux = np.array(_flux)
        log_likelihoods = np.array(_log_likelihoods).flatten()
        log_prior_values = np.array(_log_prior_values).flatten()

        # Drop anything with NaNs (e.g., failed simulation)
        mask = np.logical_and.reduce(
            (
                ~np.isnan(flux).any(axis=1),
                ~np.isnan(log_likelihoods),
                ~np.isnan(log_prior_values),
            )
        )
        samples = samples[mask]
        log_prob_samples = log_prob_samples[mask]
        flux = flux[mask]
        log_likelihoods = log_likelihoods[mask]
        log_prior_values = log_prior_values[mask]
        n = len(samples)
        print(f"Dropped {np.sum(~mask):,} invalid samples!")
        print(f"Remaining samples: {n:,} ({100 * n / len(mask):.2f}%)\n")

        # Save the results for the current job
        file_name = f"simulations-{args.job:04d}.hdf"
        print("Saving results to HDF...", end=" ", flush=True)
        save_to_hdf(
            file_path=working_dir / file_name,
            flux=flux.astype(np.float32),
            log_likelihoods=log_likelihoods.astype(np.float32),
            log_prior_values=log_prior_values.astype(np.float32),
            log_prob_samples=log_prob_samples.astype(np.float32),
            log_prob_theta_true=proposal_samples["log_prob_theta_true"],
            samples=samples.astype(np.float32),
        )
        print("Done!\n\n")

    # -------------------------------------------------------------------------
    # Stage 4: Merge the simulations from all jobs and compute the weights
    # -------------------------------------------------------------------------

    if args.stage == "merge_simulation_results" or args.stage is None:

        print(80 * "-", flush=True)
        print("(4) Merge simulation results and compute weights", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Merge the results from all simulation jobs
        print("Merging HDF files:")
        delete_after_merge = config.merge_simulation_results.delete_after_merge
        merge_hdf_files(
            target_dir=working_dir, name_pattern="simulations-*.hdf",
            output_file_path=working_dir / "simulations.hdf",
            keys=[
                "flux",
                "log_likelihoods",
                "log_prior_values",
                "log_prob_samples",
                "samples",
            ],
            singleton_keys=["log_prob_theta_true"],
            delete_after_merge=delete_after_merge,
            show_progressbar=True,
        )
        print()

        # Load the merged results
        print("Loading merged results...", end=" ", flush=True)
        merged = load_from_hdf(
            file_path=working_dir / "simulations.hdf",
            keys=[
                "flux",
                "log_likelihoods",
                "log_prior_values",
                "log_prob_samples",
                "log_prob_theta_true",
                "samples",
            ],
        )
        print("Done!")

        # Compute the importance sampling weights
        print("Computing importance sampling weights...", end=" ", flush=True)
        raw_log_weights, weights = compute_is_weights(
            log_likelihoods=merged["log_likelihoods"],
            log_prior_values=merged["log_prior_values"],
            log_probs=merged["log_prob_samples"],
        )
        merged["raw_log_weights"] = raw_log_weights.astype(np.float32)
        merged["weights"] = weights.astype(np.float32)
        print("Done!\n")

        # Compute the effective sample size and sample efficiency as well
        # as the log-evidence estimate and its standard deviation
        n_eff, sample_efficiency = compute_effective_sample_size(weights)
        log_Z, log_Z_std = compute_log_evidence(merged["raw_log_weights"])
        print(f"  Effective sample size: {n_eff:.2f}")
        print(f"  Sample efficiency:     {100 * sample_efficiency:.2f}%\n")
        print(f"  Log-evidence estimate: {log_Z:.2f} +/- {log_Z_std:.2f}\n")

        # Save the final results: full and minimized
        print("Saving results to HDF...", end=" ")
        save_to_hdf(
            file_path=working_dir / "importance_sampling_results.hdf",
            **merged,
        )
        save_to_hdf(
            file_path=working_dir / "importance_sampling_results_min.hdf",
            log_prob_theta_true=merged["log_prob_theta_true"],
            raw_log_weights=merged["raw_log_weights"],
            samples=merged["samples"],
            weights=merged["weights"],
        )
        print("Done!")

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")
