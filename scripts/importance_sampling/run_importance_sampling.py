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
)
from fm4ar.likelihoods import get_likelihood_distribution
from fm4ar.priors import get_prior
from fm4ar.simulators import get_simulator
from fm4ar.utils.hdf import load_from_hdf, merge_hdf_files, save_to_hdf
from fm4ar.utils.htcondor import (
    DAGManFile,
    HTCondorConfig,
    condor_submit_dag,
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
        help="Path to the experiment directory.",
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
        "--target-index",
        type=int,
        default=0,
        help="Index of the target spectrum to use. Default: 0.",
    )
    args = parser.parse_args()

    return args


def prepare_and_launch_dag(
    args: argparse.Namespace,
    config: ImportanceSamplingConfig,
    output_dir: Path,
) -> None:
    """
    Prepare and launch the DAGMan file for running the importance
    sampling workflow on HTCondor.

    Args:
        args: The command line arguments.
        config: The importance sampling configuration.
        output_dir: The output directory for the importance sampling run.
    """

    # Initialize a new DAGMan file
    dag = DAGManFile()

    # Add jobs for the different stages of the importance sampling workflow
    for stage, depends_on in [
        ("draw_proposal_samples", None),
        ("merge_proposal_samples", ["draw_proposal_samples"]),
        ("simulate_spectra", ["merge_proposal_samples"]),
        ("merge_simulation_results", ["simulate_spectra"]),
    ]:

        # Collect HTCondorSettings for the stage
        condor_settings: HTCondorConfig = getattr(config, stage).htcondor
        condor_settings.arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
            f"--stage {stage}",
        ]

        # For the stages that require parallel processing, add the job number
        # and the total number of parallel jobs as arguments; if we just take
        # this from the config file, things break down in non-parallel mode
        if stage in ("draw_proposal_samples", "simulate_spectra"):
            condor_settings.arguments += [
                "--job $(Process)",
                f"--n-jobs {condor_settings.queue}",
            ]

        # Create submission file
        file_path = create_submission_file(
            condor_settings=condor_settings,
            experiment_dir=output_dir,
            file_name=f"{stage}.sub",
        )

        # Add the job to the DAGMan file
        dag.add_job(
            name=stage,
            file_path=file_path,
            depends_on=depends_on,
        )

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
    print("\nRUN IMPORTANCE SAMPLING\n\n")

    # Get the command line arguments and load the importance sampling config
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Define and prepare ouput directory
    file_stem = config.target_spectrum.file_path.stem
    file_index = config.target_spectrum.index
    dir_name = f"{file_stem}__{file_index}"
    output_dir = args.experiment_dir / "importance_sampling" / dir_name
    output_dir.mkdir(exist_ok=True, parents=True)

    # -------------------------------------------------------------------------
    # If --start-submission: Create DAG file, launch job, and exit
    # -------------------------------------------------------------------------

    if args.start_submission:
        prepare_and_launch_dag(args=args, config=config, output_dir=output_dir)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "draw_proposal_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Draw samples (this comes with its own progress bar)
        theta, probs = draw_proposal_samples(args=args, config=config)

        print("\nSaving results to HDF...", end=" ", flush=True)
        save_to_hdf(
            file_path=output_dir / f"proposal-samples-{args.job:04d}.hdf",
            theta=theta,
            probs=probs,
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
        merge_hdf_files(
            target_dir=output_dir,
            name_pattern="proposal-samples-*.hdf",
            output_file_path=output_dir / "proposal-samples.hdf",
            keys=["theta", "probs"],
            singleton_keys=[],
            delete_after_merge=True,
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
        proposal_samples = load_from_hdf(
            file_path=output_dir / "proposal-samples.hdf",
            keys=["theta", "probs"],
            idx=idx,
        )
        theta = proposal_samples["theta"]
        probs = proposal_samples["probs"]

        # Load the target spectrum
        target = load_target_spectrum(
            file_path=config.target_spectrum.file_path,
            index=(
                args.target_index if args.target_index is not None
                else config.target_spectrum.index
            ),
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
            Returns the flux, the likelihood, and the prior values
            for the given `theta_i`.
            """

            # Evaluate the prior at theta_i
            # If the prior is 0, we can skip the rest (weight will be 0)
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
            likelihood = likelihood_distribution.pdf(x_i)

            return x_i, likelihood, prior_value

        # Compute spectra, likelihoods and prior values in parallel
        print("Simulating spectra (in parallel):", flush=True)
        num_cpus = get_number_of_available_cores()
        results = p_map(process_theta_i, theta, num_cpus=num_cpus, ncols=80)
        print()

        # Unpack the results from the parallel map and convert to arrays
        flux, likelihoods, prior_values = zip(*results, strict=True)
        flux = np.array(flux)
        likelihoods = np.array(likelihoods).flatten()
        prior_values = np.array(prior_values).flatten()

        # Drop everything that has a prior of 0 (i.e., is outside the bounds),
        # or where the simulation failed (i.e., the spectrum contains NaNs)
        mask = np.logical_and(prior_values > 0, ~np.isnan(flux).any(axis=1))
        theta = theta[mask]
        probs = probs[mask]
        flux = flux[mask]
        likelihoods = likelihoods[mask]
        prior_values = prior_values[mask]
        n = len(theta)
        print(f"Dropped {np.sum(~mask):,} invalid samples!")
        print(f"Remaining samples: {n:,} ({100 * n / len(mask):.2f}%)\n")

        # Save the results for the current job
        file_name = f"simulations-{args.job:04d}.hdf"
        print("Saving results to HDF...", end=" ", flush=True)
        save_to_hdf(
            file_path=output_dir / file_name,
            theta=theta.astype(np.float32),
            probs=probs.astype(np.float64),
            flux=flux.astype(np.float32),
            likelihoods=likelihoods.astype(np.float64),
            prior_values=prior_values.astype(np.float32),
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
        merge_hdf_files(
            target_dir=output_dir, name_pattern="simulations-*.hdf",
            output_file_path=output_dir / "simulations.hdf",
            keys=["theta", "probs", "flux", "likelihoods", "prior_values"],
            singleton_keys=[],
            delete_after_merge=True,
            show_progressbar=True,
        )
        print()

        # Load the merged results
        print("Loading merged results...", end=" ", flush=True)
        merged = load_from_hdf(
            file_path=output_dir / "simulations.hdf",
            keys=["theta", "probs", "flux", "likelihoods", "prior_values"],
        )
        print("Done!")

        # Compute the importance sampling weights
        print("Computing importance sampling weights...", end=" ", flush=True)
        raw_weights, weights = compute_is_weights(
            likelihoods=merged["likelihoods"],
            prior_values=merged["prior_values"],
            probs=merged["probs"],
        )
        merged["raw_weights"] = raw_weights.astype(np.float64)
        merged["weights"] = weights.astype(np.float64)
        print("Done!\n")

        # Compute the effective sample size and sample efficiency
        n_eff, sample_efficiency = compute_effective_sample_size(weights)
        print(f"  Effective sample size: {n_eff:.2f}")
        print(f"  Sample efficiency:     {100 * sample_efficiency:.2f}%\n")

        # Save the final results: full and minimized
        print("Saving results to HDF...", end=" ")
        save_to_hdf(
            file_path=output_dir / "importance_sampling_results.hdf",
            **merged,
        )
        save_to_hdf(
            file_path=output_dir / "importance_sampling_results_min.hdf",
            theta=merged["theta"],
            weights=merged["weights"],
        )
        print("Done!")

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")
