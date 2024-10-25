"""
Run importance sampling using a trained ML model: either a "proper"
posterior model (FMPE or NPE), or an unconditional flow model (e.g.,
for creating reference posteriors).
"""

import argparse
import sys
import time
from pathlib import Path
from socket import gethostname

import numpy as np
from p_tqdm import p_map

from fm4ar.importance_sampling.config import (
    ImportanceSamplingConfig,
    load_config,
)
from fm4ar.importance_sampling.proposals import draw_proposal_samples
from fm4ar.importance_sampling.utils import (
    compute_effective_sample_size,
    compute_is_weights,
    compute_log_evidence,
)
from fm4ar.likelihoods import get_likelihood_distribution
from fm4ar.priors import get_prior
from fm4ar.simulators import get_simulator
from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.torchutils.general import get_cuda_info
from fm4ar.utils.hdf import load_from_hdf, merge_hdf_files, save_to_hdf
from fm4ar.utils.htcondor import (
    DAGManFile,
    HTCondorConfig,
    check_if_on_login_node,
    condor_submit_dag,
    create_submission_file,
)
from fm4ar.utils.multiproc import get_number_of_available_cores
from fm4ar.utils.paths import expand_env_variables_in_path


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
        "--max-timeouts",
        type=int,
        default=0,
        help=(
            "Maximum number of timeouts before the job is resubmitted on a "
            "different node. Default: 0 (= no timeouts allowed)."
        ),
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
            "named `importance_sampling.yaml` (to reduce confusion). The path "
            "can either be absolute, or relative to the `importance_sampling` "
            "directory in the experiment directory."
        ),
    )
    args = parser.parse_args()

    return args


def prepare_and_launch_dag(
    args: argparse.Namespace,
    config: ImportanceSamplingConfig,
) -> None:
    """
    Prepare and launch the DAGMan file for running the importance
    sampling workflow on HTCondor.

    Args:
        args: The command line arguments.
        config: The importance sampling configuration.
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

        # For simulation jobs, we need to handle the case of "dead" nodes, that
        # is, nodes on which the simulator times out. In this case, we need to
        # fail the job with a specific exit code so that it can automatically
        # be resubmitted on a different node.
        if stage == "simulate_spectra":
            htcondor_config.retry_on_exit_code = 13
            htcondor_config.retry_on_different_node = True

        # Create submission file
        file_path = create_submission_file(
            htcondor_config=htcondor_config,
            experiment_dir=args.working_dir,
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
    file_path = args.working_dir / "0__importance_sampling.dag"
    dag.save(file_path=file_path)

    # Submit the DAGMan file to HTCondor
    condor_submit_dag(file_path=file_path, verbose=True)


def backup_target_spectrum(
    args: argparse.Namespace,
    config: ImportanceSamplingConfig,
) -> None:
    """
    Backup the target spectrum to the working directory.
    """

    # Define shortcuts
    src_file_path = expand_env_variables_in_path(
        config.target_spectrum.file_path
    )
    idx = config.target_spectrum.index

    # Save a copy of the target spectrum to the working directory
    # We only need to do this once, not for every parallel job
    file_path = args.working_dir / "target_spectrum.hdf"
    if args.job == 0 and not file_path.exists():

        # Load the target spectrum and extract the relevant index
        # This is slightly annoying because the `wlen` key requires special
        # treatment, since we don't store the wavelength grid for every sample.
        target_spectrum = load_from_hdf(file_path=src_file_path)
        for key in target_spectrum:
            if key != "wlen":
                target_spectrum[key] = target_spectrum[key][idx]
            else:
                target_spectrum[key] = target_spectrum[key].flatten()

        # Save the target spectrum to the working directory
        save_to_hdf(file_path=file_path, **target_spectrum)


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nRUN IMPORTANCE SAMPLING\n")

    # Get the command line arguments and define shortcuts
    args = get_cli_arguments()

    # Resolve the working directory in case it is not absolute
    # We need to overwrite args because some functions expect args.working_dir
    # to be an absolute path. Not sure if there is a cleaner way to do this?
    if not args.working_dir.is_absolute():
        args.working_dir = expand_env_variables_in_path(
            Path(args.experiment_dir)
            / "importance_sampling"
            / args.working_dir
        )

    # Make sure the working directory exists before we proceed
    if not args.working_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.working_dir}")

    # Ensure that we do not run compute-heavy jobs on the login node
    check_if_on_login_node(start_submission=args.start_submission)
    print("Running on host:", gethostname(), "\n", flush=True)

    # Load the importance sampling config
    config = load_config(experiment_dir=args.working_dir)

    # -------------------------------------------------------------------------
    # If --start-submission: Create DAG file, launch job, and exit
    # -------------------------------------------------------------------------

    if args.start_submission:
        backup_target_spectrum(args=args, config=config)
        prepare_and_launch_dag(args=args, config=config)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Stage 1: Draw samples from the proposal distribution
    # -------------------------------------------------------------------------

    if args.stage == "draw_proposal_samples" or args.stage is None:

        print(80 * "-", flush=True)
        print("(1) Draw samples from proposal distribution", flush=True)
        print(80 * "-" + "\n", flush=True)

        # Check if the output file or the merged proposals file already exists
        output_file_path = (
            args.working_dir / f"proposal-samples-{args.job:04d}.hdf"
        )
        if output_file_path.exists():
            print(f"{output_file_path.name} exists already, skipping!\n")
        elif (args.working_dir / "proposal-samples.hdf").exists():
            print("Merged proposal samples exist already, skipping!\n")

        # Otherwise, we need to draw the proposal samples
        else:

            # Document the CUDA setup
            print("CUDA information:")
            for key, value in get_cuda_info().items():
                print(f"  {key + ':':<16}{value}")
            print()

            # Draw samples (this comes with its own progress bar)
            results = draw_proposal_samples(args=args, config=config)

            # Convert some arrays to float32 to save space
            for key in ("samples", "log_prob_samples"):
                results[key] = results[key].astype(np.float32)

            print("\nSaving results to HDF...", end=" ", flush=True)
            save_to_hdf(
                file_path=output_file_path,
                samples=results["samples"],
                log_prob_samples=results["log_prob_samples"],
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
            target_dir=args.working_dir,
            name_pattern="proposal-samples-*.hdf",
            output_file_path=args.working_dir / "proposal-samples.hdf",
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

        # Construct path for the expected output file
        # If the file already exists, we don't need to simulate the spectra.
        # This can happen if another part of the DAG has failed (e.g., too
        # many timeouts) and the job is resubmitted.
        output_file_path = args.working_dir / f"simulations-{args.job:04d}.hdf"
        if output_file_path.exists():
            print(f"{output_file_path.name} exists already, skipping!\n")
        elif (args.working_dir / "simulations.hdf").exists():
            print("Merged simulation results exist already, skipping!\n")

        # Otherwise, we need to simulate the spectra
        # TODO: Obviously, it would be cleaner to wrap all of this this in a
        #   separate function, but this isn't trivial because of the parallel
        #   processing which requires to pickle the simulator...
        else:

            # Construct the slice of indices for the current job: The current
            # job will process every `n_jobs`-th sample from the proposal
            # samples, starting at an index of `job`.
            idx = slice(args.job, None, args.n_jobs)

            # Print some information about the number of samples to process
            n_total = config.draw_proposal_samples.n_samples
            n_for_job = len(np.arange(args.job, n_total, args.n_jobs))
            print(f"Total number of samples:             {n_total:,}")
            print(f"Number of samples for current job:   {n_for_job:,}")
            print()

            # Load the theta samples and probabilities and unpack them
            # We also load the log-probability of the ground truth theta value
            proposal_samples = load_from_hdf(
                file_path=args.working_dir / "proposal-samples.hdf",
                keys=["samples", "log_prob_samples", "log_prob_theta_true"],
                idx=idx,
            )
            samples = proposal_samples["samples"]
            log_prob_samples = proposal_samples["log_prob_samples"]

            # Sanity check: Are there any duplicate samples? (This can happen
            # if there are issues with the random seed for each proposal job)
            n_duplicates = len(np.unique(samples, axis=0)) - len(samples)
            if n_duplicates > 0:
                raise ValueError(
                    f"Found {n_duplicates:,} duplicates in the proposal!"
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
                error_bars=target["error_bars"],
            )

            # Set up a counter for the number of simulator timeouts.
            # The upper limit is defined in the command line arguments.
            n_timeouts = 0

            # Define a function that processes a single `theta_i`
            def process_theta_i(
                theta_i: np.ndarray,
            ) -> tuple[np.ndarray, float, float]:
                """
                Returns the flux, the log-likelihood, and the log-prior
                values for the given `theta_i`.
                """

                # Evaluate the prior at theta_i
                # If the prior is 0, we don't really need to run the simulator
                # since the importance sampling weight will be 0 anyway.
                if (prior_value := prior.evaluate(theta_i)) <= 0:
                    return np.full(n_bins, np.nan), -np.inf, -np.inf
                log_prior_value = np.log(prior_value)

                # Simulate the spectrum that belongs to theta_i
                # If we get `None` here, it means the simulator has timed out,
                # which usually only happens when running on a node that is
                # experiencing some issues. If the number of timeouts across
                # all parallel processes exceeds a certain threshold, we raise
                # an error, which will trigger a resubmission of the job on a
                # different node. We cannot directly call `sys.exit()` inside
                # this function because this will cause the script to hang.
                # Instead, we need to raise an exception and handle the exit
                # code in __main__.
                result = simulator(theta_i)
                if result is None:

                    # Increment the counter and check if limit is reached
                    global n_timeouts  # access the counter defined outside
                    n_timeouts += 1
                    if n_timeouts > args.max_timeouts:
                        raise RuntimeError("Too many timeouts!")

                    # Print a debug message to stderr and return NaNs
                    print(
                        f"Simulator timed out! ({n_timeouts=})",
                        file=sys.stderr,
                        flush=True,
                    )
                    return np.full(n_bins, np.nan), -np.inf, -np.inf

                else:
                    _, flux = result

                # Compute the log-likelihood
                # We use the log-likelihood to avoid numerical issues, because
                # the likelihood can take on values on the order of 10^-1000.
                # In cases where the `flux` contains NaNs, the log-likelihood
                # will also be NaN, which will result in a weight of 0.
                log_likelihood = likelihood_distribution.logpdf(flux)
                if np.isnan(log_likelihood):
                    print("NaN in loglikelihood!", file=sys.stderr, flush=True)
                    return np.full(n_bins, np.nan), -np.inf, -np.inf

                return flux, log_likelihood, log_prior_value

            # Compute spectra, likelihoods and prior values in parallel
            # If we encounter too many timeouts, we exit with code 13, which
            # will trigger a resubmission of the job on a different node.
            print("Simulating spectra (in parallel):", flush=True)
            num_cpus = get_number_of_available_cores()
            try:
                results = p_map(
                    process_theta_i,
                    samples,
                    num_cpus=num_cpus,
                    ncols=80,
                )
            except RuntimeError as e:
                if "Too many timeouts!" in str(e):
                    sys.exit(13)
                else:
                    raise e
            print()

            # Unpack the results from the parallel map and convert to arrays
            _flux, _log_likelihoods, _log_prior_values = zip(
                *results,
                strict=True,
            )
            flux = np.array(_flux)
            log_likelihoods = np.array(_log_likelihoods).flatten()
            log_prior_values = np.array(_log_prior_values).flatten()

            # Save the results for the current job
            print("Saving results to HDF...", end=" ", flush=True)
            save_to_hdf(
                file_path=output_file_path,
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
            target_dir=args.working_dir,
            name_pattern="simulations-*.hdf",
            output_file_path=args.working_dir / "simulations.hdf",
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
            file_path=args.working_dir / "simulations.hdf",
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

        # Compute the effective sample size, the sampling efficiency and the
        # simulation_efficiency, as well as the log-evidence and its std
        (
            n_eff,
            sampling_efficiency,
            simulation_efficiency
        ) = compute_effective_sample_size(
            weights=weights,
            log_prior_values=merged["log_prior_values"],
        )
        log_Z, log_Z_std = compute_log_evidence(merged["raw_log_weights"])
        print(f"  Effective sample size: {n_eff:.2f}")
        print(f"  Sampling efficiency:   {100 * sampling_efficiency:.2f}%\n")
        print(f"  Simulation efficiency: {100 * simulation_efficiency:.2f}%\n")
        print(f"  Log-evidence estimate: {log_Z:.3f} +/- {log_Z_std:.3f}\n")

        # Add these values to the merged results
        merged["n_eff"] = np.array(n_eff)
        merged["sampling_efficiency"] = np.array(sampling_efficiency)
        merged["simulation_efficiency"] = np.array(simulation_efficiency)
        merged["log_evidence"] = np.array(log_Z)
        merged["log_evidence_std"] = np.array(log_Z_std)

        # Save the full results
        print("Saving full results to HDF...", end=" ")
        save_to_hdf(file_path=args.working_dir / "results.hdf", **merged)
        print("Done!")

        # Drop some quantities that are usually not needed for downstream
        # analysis and save a minimized version
        print("Saving minimized results to HDF...", end=" ")
        del merged["flux"]
        del merged["log_prior_values"]
        del merged["log_prob_samples"]
        del merged["log_likelihoods"]
        save_to_hdf(file_path=args.working_dir / "results.min.hdf", **merged)
        print("Done!")

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\n\nThis took {time.time() - script_start:.2f} seconds.\n")
